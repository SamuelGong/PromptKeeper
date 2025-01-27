import os
import re
import gc
import time
import torch
import shutil
import pickle
import logging
import subprocess

from datasets import Dataset
from transformers import T5Tokenizer
from promptkeeper.llm_related import get_text_generation
from promptkeeper.defense import get_defended_model_response
from promptkeeper.utils import read_file, save_customized_log
from promptkeeper.evaluate_attack import evaluate_an_attack, fast_analysis


output2prompt_query_filenames = [
    'questions_to_ask.txt',
    'scenarios_to_use.txt',
    'describe_yourself.txt',
    'cmp_w_chatgpt.txt'
]


def reconstruct_w_output2prompt(repetition_dir, final_response_list,
                                possible_reusable_result_path=None):
    new_text_to_save = [
        ['text', 'INPUT TAKEN BY OUTPUT2PROMPT'],
        ['text', '\n'.join(final_response_list)]
    ]

    if (possible_reusable_result_path is not None and
            os.path.isfile(possible_reusable_result_path)):  # optimize for parameterized methods
        with open(possible_reusable_result_path, 'r') as fin:
            reconstructed_system_prompt = fin.read()
    else:
        project_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(project_dir)
        output2prompt_dir = os.path.join(parent_dir, 'output2prompt')
        if not os.path.isdir(output2prompt_dir):
            raise ValueError(f"{output2prompt_dir} not found. Please first prepare the repo for "
                             f"output2prompt in the parent directory of this project (details in README.md)!")

        # to release memory pressure
        llm_related._transformers_cached_model_id = ""
        del llm_related._transformers_cached_model
        del llm_related._transformers_cached_pipeline
        # only these two "del"'s combined will work
        gc.collect()
        torch.cuda.empty_cache()
        llm_related._transformers_cached_model = None
        llm_related._transformers_cached_pipeline = None

        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        dummy_prompt = "dummy"
        dummy_prompt_tokenized = tokenizer(
            dummy_prompt,
            padding='max_length',
            max_length=256  # no truncation here
        )["input_ids"]
        # print(final_response_list)
        # exit(0)
        final_response_list_tokenized = tokenizer(
            final_response_list,
            padding='max_length',
            max_length=32,
            truncation=True
        )["input_ids"]
        data = {
            "system_prompt": [dummy_prompt_tokenized],
            "result_list": [final_response_list_tokenized],
            "names": ["GPT"],
            "questions": [[]]
        }
        input_data_path = os.path.join(repetition_dir, 'input_for_output2prompt')
        input_dataset = Dataset.from_dict(data)
        input_dataset.save_to_disk(
            dataset_path=input_data_path
        )

        output2prompt_main_path = os.path.join(output2prompt_dir, 'main.py')
        absolute_input_data_path = os.path.join(os.getcwd(), input_data_path)
        command = ['python', f'{output2prompt_main_path}', 'test', 'system_prompts',
                   f"{absolute_input_data_path}"]
        subprocess.run(command, cwd=output2prompt_dir)
        shutil.rmtree(absolute_input_data_path)

        output_data_path = os.path.join(output2prompt_dir, 'temp.txt')
        with open(output_data_path, 'r') as f:
            reconstructed_system_prompt = f.read()
        os.remove(output_data_path)

        if possible_reusable_result_path is not None:
            with open(possible_reusable_result_path, 'w') as fout:
                fout.write(reconstructed_system_prompt)

    new_text_to_save += [
        ['text', 'RECONSTRUCTED SYSTEM PROMPT BY OUTPUT2PROMPT'],
        ['text', reconstructed_system_prompt]
    ]
    # print(reconstructed_system_prompt)
    return reconstructed_system_prompt, new_text_to_save


def output2prompt_extract_from_response(defense_mode_params, final_response):
    lines = final_response.split('\n')
    pattern = re.compile(r'^\d+[.:]')

    if defense_mode_params.name == "empty":
        result = [""] * 16  # TODO: avoid hard-coding
    else:
        result = []
        for line in lines:
            if pattern.match(line):
                result.append(line)
    return result


def regular_query_attack(config, defense_mode_params, system_prompt, system_prompt_name,
                         prompt_dir, raw_res_for_save_path, raw_res_for_save):
    defense_mode = defense_mode_params.name
    separate_line = '-' * 10

    if config.regular_query_attack.name == "output2prompt":
        attempt_dict = {
            1: "English", 2: "Chinese", 3: "German",
            4: "Russian", 5: "Spanish", 6: "Swedish"
        }
        user_query_name_list = [
            "describe_yourself", "questions_to_ask", "scenarios_to_use", "cmp_w_chatgpt",
        ]
        query_dir = "user_query/regular/output2prompt"

        for ri in range(1, len(attempt_dict) + 1):
            attempt = attempt_dict[ri]
            if str(ri) in raw_res_for_save[defense_mode][system_prompt_name]:
                logging.info(f"\t\tAttempt for {attempt} "
                             f"({ri}/{len(attempt_dict)}) "
                             f"has been done before.")
                continue
            start_time = time.perf_counter()
            repetition_dir = os.path.join(prompt_dir, f"{ri}")
            os.makedirs(repetition_dir, exist_ok=True)
            text_to_save = []
            final_response_list = []

            # Step One: Obtaining responses
            reconstructed_text_path_prefix = ""   # tailored for parameterized methods
            for query_id in range(1, len(user_query_name_list) + 1):
                user_query_name = user_query_name_list[query_id - 1] + f"_{attempt.lower()}"
                user_query_path = os.path.join(query_dir, user_query_name + ".txt")
                text_to_save.append(['text', f'{separate_line}OBTAINING RESPONSE WITH REGULAR QUERY '
                                             f'{user_query_name.upper()}{separate_line}'])

                final_response_path = os.path.join(repetition_dir, f'{user_query_name}.txt')
                final_response_meta_path = os.path.join(repetition_dir, f'{user_query_name}.pkl')
                if os.path.isfile(final_response_meta_path):
                    logging.info(f"\t\tLoading existing response with {user_query_name} "
                                 f"({query_id}/{len(user_query_name_list)}).")
                    with open(final_response_path, 'r') as fin:
                        final_response = fin.read()
                    with open(final_response_meta_path, 'rb') as fin:
                        new_text_to_save = pickle.load(fin)

                    text_to_save += new_text_to_save
                    extracted_final_response = output2prompt_extract_from_response(
                        defense_mode_params, final_response
                    )
                else:
                    if "mll_test_regen" in defense_mode:  # optimize for these parameterized methods
                        defense_mode_params.system_prompt_name = system_prompt_name

                        parent_dir = os.path.dirname(raw_res_for_save_path)
                        raw_text_dir = os.path.join(parent_dir, "raw_response", "mll_test_regen",
                                                    system_prompt_name, f"{ri}_{user_query_name}")
                        os.makedirs(raw_text_dir, exist_ok=True)
                        defense_mode_params.raw_text_dir = raw_text_dir

                    user_query = read_file(user_query_path)

                    retry_time = 5
                    counter = 0
                    while counter < retry_time:
                        counter += 1
                        new_text_to_save = []
                        logging.info(f"\t\tObtaining response with {user_query_name} "
                                     f"({query_id}/{len(output2prompt_query_filenames)}) "
                                     f"for the {counter}-th time...")
                        new_text_to_save.append(['text', f'{counter}-th GENERATION ATTEMPT'.upper()])

                        final_response, _new_text_to_save = get_defended_model_response(
                            model=config.model,
                            embed_model=config.embed_model,
                            defense_mode_params=defense_mode_params,
                            system_prompt=system_prompt,
                            user_query=user_query
                        )
                        new_text_to_save += _new_text_to_save

                        translation_sp = (f"Translate the following text into English using "
                                          f"{config.regular_query_attack.translate_model}:")
                        new_text_to_save.append(['text', 'SYSTEM PROMPT FOR LANGUAGE TRANSLATION:'])
                        new_text_to_save.append(['text', translation_sp])
                        translation_q = f"[The Start of Text]\n{final_response}\n[The End of Text]"
                        new_text_to_save.append(['text', 'USER QUERY FOR LANGUAGE TRANSLATION:'])
                        new_text_to_save.append(['text', translation_q])

                        # exception one
                        try:
                            logging.info(f"\t\t\tTranslating final response...")
                            translated_final_response = get_text_generation(
                                model_id=config.regular_query_attack.translate_model,
                                user_query=translation_q,
                                system_prompt=translation_sp
                            )
                        # except openai.BadRequestError as e:
                        except Exception as e:
                            logging.info(f"\t\t\tResponse translation failed due to {type(e)}: {e}.")
                            new_text_to_save.append(['text', f'RESPONSE TRANSLATION '
                                                             f'FAILED DUE TO: {str(e).upper()}'])
                            continue

                        if translated_final_response is None:
                            logging.info(f"\t\t\tResponse translation result is None.")
                            new_text_to_save.append(['text', f'RESPONSE TRANSLATION RESULT IS NONE'])
                            continue

                        new_text_to_save.append(['text', 'TRANSLATED FINAL RESPONSE:'])
                        new_text_to_save.append(['text', translated_final_response])
                        extracted_final_response = output2prompt_extract_from_response(
                            defense_mode_params, translated_final_response
                        )

                        # exception two
                        # expected 16; preferrably at least a half
                        min_num_samples = 8  # TODO: avoid hard-coding
                        if len(extracted_final_response) < min_num_samples:
                            logging.info(f"\t\t\tExtracted {len(extracted_final_response)}"
                                         f"<{min_num_samples} response.")
                            new_text_to_save.append(['text',
                                                     f'EXTRACTED {len(extracted_final_response)}'
                                                     f'<{min_num_samples} RESPONSE.'])
                            continue
                        break
                    if counter >= retry_time:
                        logging.info(f"\t\tFailed to obtain response with {user_query_name} "
                                     f"({query_id}/{len(output2prompt_query_filenames)}) "
                                     f"after {retry_time} times. Skip this query.")
                        continue

                    new_text_to_save.append(['text', 'EXTRACTED FINAL RESPONSE:'])
                    new_text_to_save.append(['text', '\n'.join(extracted_final_response)])
                    with open(final_response_path, "w") as fout:
                        fout.write(final_response)
                    with open(final_response_meta_path, "wb") as fout:
                        pickle.dump(new_text_to_save, fout)

                text_to_save += new_text_to_save
                final_response_list += extracted_final_response

                if "mll_test_regen" in defense_mode:  # optimize for these parameterized methods
                    has_regenerated = 0  # without the response was regenerated
                    for item in new_text_to_save:
                        if "REGENERATING WITHOUT" in item[1]:
                            has_regenerated = 1
                            break
                    if has_regenerated:
                        reconstructed_text_path_prefix += f"_{query_id}o"
                    else:
                        reconstructed_text_path_prefix += f"_{query_id}w"
            text_to_save.append(['text', f'{separate_line}{separate_line}{separate_line}'])

            # Step 2: Utilizing responses
            if "mll_test_regen" in defense_mode:  # optimize for these parameterized methods
                parent_dir = os.path.dirname(raw_res_for_save_path)
                reconstructed_text_dir = os.path.join(parent_dir, "raw_reconstruction",
                                                      "mll_test_regen", system_prompt_name)
                os.makedirs(reconstructed_text_dir, exist_ok=True)
                possible_reusable_result_path = os.path.join(
                    reconstructed_text_dir,
                    f"{ri}{reconstructed_text_path_prefix}.txt"
                )
            else:
                possible_reusable_result_path = None

            retry_time = 5
            counter = 0
            while counter < retry_time:
                counter += 1
                logging.info(f"\t\tReconstructing system prompt using output2prompt "
                             f"for the {counter}-th time...")
                reconstructed_system_prompt, new_text_to_save = reconstruct_w_output2prompt(
                    repetition_dir=repetition_dir,
                    final_response_list=final_response_list,
                    possible_reusable_result_path=possible_reusable_result_path
                )
                # exception one
                try:
                    attack_eval_res, new_text_to_save_2 = evaluate_an_attack(
                        config=config,
                        predicted_text=reconstructed_system_prompt,
                        system_prompt=system_prompt,
                        # output2prompt already takes English stuffs as input
                        text_must_be_english=True
                    )
                except Exception as e:
                    logging.info(f"\t\t\tAttack evaluation failed due to type{e}: {e}.")
                    continue
                break
            if counter >= retry_time:
                logging.info(f"\t\tFailed to reconstruct system prompt "
                             f"after {retry_time} times. Skip this attempt.")
                continue

            text_to_save += new_text_to_save
            text_to_save += new_text_to_save_2
            log_path = os.path.join(repetition_dir, f"log.txt")
            save_customized_log(text_to_save, file_prefix=log_path, need_postfix=False)

            raw_res_for_save[defense_mode][system_prompt_name][str(ri)] = attack_eval_res  # str() is important
            # periodically save by overwriting
            pickle.dump(raw_res_for_save, open(raw_res_for_save_path, 'wb'))

            end_time = time.perf_counter()
            duration = end_time - start_time
            logging.info(f"\t\tAttempt {ri}/{len(attempt_dict)} "
                         f"done in {round(duration, 2)}s.")
    else:
        raise NotImplementedError


def adversarial_query_attack(config, defense_mode_params, system_prompt, system_prompt_name,
                             prompt_dir, raw_res_for_save_path, raw_res_for_save):
    defense_mode = defense_mode_params.name

    user_query_filenames = [file for file in os.listdir(config.user_query_dir)
                            if file.endswith('.txt')]
    for query_id, user_query_filename in enumerate(user_query_filenames):
        user_query_name = user_query_filename.split('/')[-1].split('.')[0]
        if user_query_name not in raw_res_for_save[defense_mode][system_prompt_name]:
            raw_res_for_save[defense_mode][system_prompt_name][user_query_name] = {}
        user_query_path = os.path.join(config.user_query_dir, user_query_filename)
        user_query = read_file(user_query_path)
        logging.info(f"\t\tAttacking with user query {user_query_name} "
                     f"({query_id + 1}/{len(user_query_filenames)}).")

        for ri in range(1, config.num_resp_per_query + 1):
            if str(ri) in raw_res_for_save[defense_mode][system_prompt_name][user_query_name]:
                logging.info(f"\t\t\tAttempt {ri}/{config.num_resp_per_query} has been done before.")
                continue
            start_time = time.perf_counter()
            text_to_save = []

            if "mll_test_regen" in defense_mode:  # optimize for these parameterized methods
                defense_mode_params.system_prompt_name = system_prompt_name

                parent_dir = os.path.dirname(raw_res_for_save_path)
                raw_text_dir = os.path.join(parent_dir, "raw_response", "mll_test_regen",
                                            system_prompt_name, f"{user_query_name}_{ri}")
                os.makedirs(raw_text_dir, exist_ok=True)
                defense_mode_params.raw_text_dir = raw_text_dir
            final_response, new_text_to_save = get_defended_model_response(
                model=config.model,
                embed_model=config.embed_model,
                defense_mode_params=defense_mode_params,
                system_prompt=system_prompt,
                user_query=user_query
            )
            text_to_save += new_text_to_save

            retry_time = 5
            counter = 0
            while counter < retry_time:
                counter += 1
                try:
                    attack_eval_res, new_text_to_save = evaluate_an_attack(
                        config=config,
                        predicted_text=final_response,
                        system_prompt=system_prompt
                    )
                except Exception as e:
                    logging.info(f"\t\t\t\tAttack evaluation failed due to type{e}: {e}.")
                    continue
                break
            if counter >= retry_time:
                logging.info(f"\t\t\tFailed to evaluate attack "
                             f"after {retry_time} times. Skip this attempt.")
                continue
            text_to_save += new_text_to_save

            log_path = os.path.join(prompt_dir, f"{user_query_name}_{ri}.txt")
            save_customized_log(text_to_save, file_prefix=log_path, need_postfix=False)
            raw_res_for_save[defense_mode][system_prompt_name][user_query_name][str(ri)] \
                = attack_eval_res  # str() is important
            # periodically save by overwriting
            pickle.dump(raw_res_for_save, open(raw_res_for_save_path, 'wb'))

            end_time = time.perf_counter()
            duration = end_time - start_time
            logging.info(f"\t\t\tAttempt {ri}/{config.num_resp_per_query} done in {round(duration, 2)}s.")

        _, inner_res_for_print = fast_analysis(
            raw_res_for_save[defense_mode][system_prompt_name][user_query_name])
        logging.info(f"\t\tBest attack result for user query {user_query_name}: {inner_res_for_print}")
        _, inner_res_for_print_mean = fast_analysis(
            raw_res_for_save[defense_mode][system_prompt_name][user_query_name], mode="mean")
        logging.info(f"\t\tMean attack result for user query {user_query_name}: {inner_res_for_print_mean}")

    _, outer_res_for_print = fast_analysis(raw_res_for_save[defense_mode][system_prompt_name])
    logging.info(f"\tBest attack result against {system_prompt_name} "
                 f"for defense mode {defense_mode}: {outer_res_for_print}")
    _, outer_res_for_print_mean = fast_analysis(raw_res_for_save[defense_mode][system_prompt_name], mode="mean")
    logging.info(f"\tMean attack result against {system_prompt_name} "
                 f"for defense mode {defense_mode}: {outer_res_for_print_mean}")
