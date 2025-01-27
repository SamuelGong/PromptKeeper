import os
import time
import math
import openai
import pickle
import logging
from munch import DefaultMunch

from promptkeeper.utils import read_file, save_customized_log
from promptkeeper.hypothesis_test import fit_distribution
from promptkeeper.evaluate_attack import calc_embed_dist, fast_analysis
from promptkeeper.hypothesis_test import get_likelihood_ratio
from promptkeeper.llm_related import get_text_generation, get_text_embedding
from promptkeeper.likelihood_test import get_likelihood_ratio_given_significance


def before_get_upperbound_for_cmp_embed_regen(config, defense_mode_params):
    if defense_mode_params.embed_model is not None:
        # this should prevail
        embed_model = defense_mode_params.embed_model
    else:
        embed_model = config.embed_model

    defense_mode_params.upperbound_compute_params\
        .save_dir = "upperbound_computed"
    defense_mode_params.upperbound_compute_params\
        .system_prompt_dir = config.system_prompt_dir
    if defense_mode_params.upperbound_compute_params\
        .user_query_dir is None:
        defense_mode_params.upperbound_compute_params\
            .user_query_dir = config.user_query_dir

    system_prompt_language = "English"  # for translation
    if config.system_prompt_language is not None:
        system_prompt_language = config.system_prompt_language
    defense_mode_params.upperbound_compute_params\
        .system_prompt_language = system_prompt_language
    defense_mode_params.upperbound_compute_params\
        .translate_model = config.translate_model
    return defense_mode_params, embed_model


def get_upperbound_for_cmp_embed_regen(model, embed_model, upperbound_compute_params):
    begin_time = time.perf_counter()
    save_dir = upperbound_compute_params.save_dir
    os.makedirs(save_dir, exist_ok=True)
    system_prompt_dir = upperbound_compute_params.system_prompt_dir
    system_prompt_filenames = [
        file for file in os.listdir(system_prompt_dir)
        if file.endswith('.txt')
    ]

    res_path = os.path.join(save_dir, f'{embed_model}_res.pkl')
    if os.path.isfile(res_path):
        with open(res_path, 'rb') as fin:
            res = pickle.load(fin)
        logging.info(f"[Upperbound] Previous results for model {model} "
                     f"loaded from {res_path}")
    else:
        res = {}
    if model not in res:
        res[model] = {}

    mode = upperbound_compute_params.mode
    for prompt_id, system_prompt_filename in enumerate(system_prompt_filenames):
        system_prompt_name = system_prompt_filename.split('/')[-1].split('.')[0]
        system_prompt_path = os.path.join(system_prompt_dir, system_prompt_filename)
        system_prompt = read_file(system_prompt_path)
        if system_prompt_name not in res[model]:
            res[model][system_prompt_name] = {}
        prompt_dir = os.path.join(save_dir, system_prompt_name)
        os.makedirs(prompt_dir, exist_ok=True)
        logging.info(f"[Upperbound]\tUsing system prompt {system_prompt_name} "
                     f"({prompt_id + 1}/{len(system_prompt_filenames)}).")

        user_query_dir = upperbound_compute_params.user_query_dir
        user_query_filenames = [
            file for file in os.listdir(user_query_dir)
            if file.endswith('.txt')
        ]
        for query_id, user_query_filename in enumerate(user_query_filenames):
            user_query_name = user_query_filename.split('/')[-1].split('.')[0]
            if user_query_name not in res[model][system_prompt_name]:
                res[model][system_prompt_name][user_query_name] = {}
            user_query_path = os.path.join(user_query_dir, user_query_filename)
            user_query = read_file(user_query_path)
            logging.info(f"[Upperbound]\t\tUsing user query {user_query_name} "
                         f"({query_id + 1}/{len(user_query_filenames)}).")

            num_resp_per_query = upperbound_compute_params.num_resp_per_query
            for ri in range(1, num_resp_per_query + 1):
                if mode in ["no_prompt"]:
                    _config = {
                        "model": model
                    }
                    _config = DefaultMunch.fromDict(_config)
                    _defense_mode_params = {
                        "name": "no_prompt"
                    }
                    _defense_mode_params = DefaultMunch.fromDict(_defense_mode_params)
                else:
                    raise NotImplementedError

                if str(ri) in res[model][system_prompt_name][user_query_name]:
                    logging.info(f"[Upperbound]\t\t\tAttempt {ri}/{num_resp_per_query} "
                                 f"under mode {mode} has been done before.")
                    continue

                final_response, text_to_save = get_defended_model_response(
                    model=model,
                    defense_mode_params=_defense_mode_params,
                    system_prompt=system_prompt,
                    user_query=user_query
                )

                system_prompt_language = upperbound_compute_params.system_prompt_language
                translation_sp = f"Translate the following text into {system_prompt_language}."
                text_to_save.append(['text', 'SYSTEM PROMPT FOR LANGUAGE TRANSLATION:'])
                text_to_save.append(['text', translation_sp])
                translation_q = f"[The Start of Text]\n{final_response}\n[The End of Text]"
                text_to_save.append(['text', 'USER QUERY FOR LANGUAGE TRANSLATION:'])
                text_to_save.append(['text', translation_q])

                translated_response = get_text_generation(
                    model_id=upperbound_compute_params.translate_model,
                    user_query=translation_q,
                    system_prompt=translation_sp
                )
                text_to_save.append(['text', 'TRANSLATED FINAL RESPONSE:'])
                text_to_save.append(['text', translated_response])

                system_prompt_embedding = get_text_embedding(
                    embed_model_id=embed_model,
                    text=system_prompt
                )
                translated_response_embedding = get_text_embedding(
                    embed_model_id=embed_model,
                    text=translated_response
                )
                cosine_similarity = calc_embed_dist(
                    embedding1=system_prompt_embedding,
                    embedding2=translated_response_embedding
                )["cosine similarity"]
                if math.isnan(cosine_similarity):
                    raise ValueError(f"[Upperbound]\t\t\tCosine similarity is Nan:\n"
                                     f"system_prompt_embedding ({system_prompt_embedding.shape}): "
                                     f"{system_prompt_embedding}\n"
                                     f"final_response_embedding ({translated_response_embedding.shape}): "
                                     f"{translated_response_embedding}")
                res[model][system_prompt_name][user_query_name][str(ri)] \
                    = {'cosine similarity': cosine_similarity}
                # periodically save by overwriting
                pickle.dump(res, open(res_path, 'wb'))
                log_path = os.path.join(prompt_dir, f"{user_query_name}_{ri}.txt")
                save_customized_log(text_to_save, file_prefix=log_path, need_postfix=False)

                logging.info(f"[Upperbound]\t\t\tAttempt {ri}/{num_resp_per_query} "
                             f"under mode {mode} done.")

    analysis_res, _ = fast_analysis(
        data_dict=res[model],
        metric_to_analyze=["cosine similarity"],
        mode="mean"
    )
    upperbound = analysis_res["cosine similarity"]["mean"]
    end_time = time.perf_counter()
    duration = end_time - begin_time
    logging.info(f"[Upperbound] Upperbound {upperbound} for model {model} "
                 f"with cmp_embed_regen loaded in {round(duration, 2)}s.")
    return upperbound


def get_hypothesis_threshold_for_mll_test_regen(model_name, system_prompt,
                                                defense_mode_params, significance):
    flattened_model_name = model_name.replace('/', '-')
    fitted_params_filename = f'fit_result.pkl'

    system_prompt_name = defense_mode_params.system_prompt_name
    assert system_prompt_name is not None
    fitted_params_dir = os.path.join('system_prompt_meta', system_prompt_name,
                                     flattened_model_name)
    fitted_params_path = os.path.join(fitted_params_dir, fitted_params_filename)
    if not os.path.isfile(fitted_params_path):  # need to fit first
        begin_time = time.perf_counter()
        logging.info(f"\tFitting distribution for {model_name} under {system_prompt_name}.")

        fit_config = DefaultMunch.fromDict({
            'model': model_name,
            'num_responses_per_query': 1,
            'num_queries_per_case': 10,
            'dist': 'norm'
        })
        questions_dir = None
        if defense_mode_params.questions_dir is not None:
            questions_dir = defense_mode_params.questions_dir
            questions_dir = os.path.join('system_prompt_meta', questions_dir,
                                         flattened_model_name)
        fit_answer_generate_kwargs = None
        if defense_mode_params.fit_answer_generate_kwargs is not None:
            fit_answer_generate_kwargs = defense_mode_params.fit_answer_generate_kwargs
        fit_distribution(
            config=fit_config,
            system_prompt=system_prompt,
            saving_dir=fitted_params_dir,
            questions_dir=questions_dir,
            answer_generate_kwargs=fit_answer_generate_kwargs
        )
        end_time = time.perf_counter()
        duration = end_time - begin_time
        logging.info(f"\tFitted for {model_name} under {system_prompt_name} in {round(duration, 2)}s")
    with open(fitted_params_path, 'rb') as fin:
        stats_data = pickle.load(fin)
        logging.info(f"\tLoaded fitted distribution for {model_name} under {system_prompt_name}.")

    assumed_dist = stats_data['dist']
    zero_dist_args = stats_data["zero"]["fitted_params"]
    other_dist_args = stats_data["other"]["fitted_params"]
    fitted_params = zero_dist_args + other_dist_args

    hypothesis_test_threshold = get_likelihood_ratio_given_significance(
        dist=assumed_dist,
        fitted_params=fitted_params,
        significance=significance,
    )
    return (hypothesis_test_threshold, assumed_dist,
            zero_dist_args, other_dist_args)


def get_defended_model_response(model, defense_mode_params, system_prompt, user_query,
                                embed_model=None, generate_kwargs=None):
    if defense_mode_params.name in ["direct", "system_prompt_ext"]:
        text_to_save = [
            ['text', 'SYSTEM PROMPT:'],
            ['text', system_prompt],
        ]
        if defense_mode_params.name == "system_prompt_ext":
            content_to_append = read_file(defense_mode_params.path_to_content_to_append)
            system_prompt += '\n' + content_to_append
            text_to_save += [
                ['text', 'NEW SYSTEM PROMPT AFTER EXTENSION:'],
                ['text', system_prompt],
            ]

        final_response = get_text_generation(
            model_id=model,
            user_query=user_query,
            system_prompt=system_prompt,
            generate_kwargs=generate_kwargs
        )
        text_to_save += [
            ['text', 'USER QUERY:'],
            ['text', user_query],
            ['text', 'FINAL RESPONSE:'],
            ['text', final_response]
        ]
    elif defense_mode_params.name in ["no_prompt"]:
        final_response = get_text_generation(
            model_id=model,
            user_query=user_query,
            generate_kwargs=generate_kwargs
        )
        text_to_save = [
            ['text', 'USER QUERY:'],
            ['text', user_query],
            ['text', 'FINAL RESPONSE:'],
            ['text', final_response]
        ]
    elif defense_mode_params.name == "empty":
        text_to_save = [
            ['text', 'SYSTEM PROMPT:'],
            ['text', system_prompt],
            ['text', 'USER QUERY:'],
            ['text', user_query],
            ['text', 'EMPTY FINAL RESPONSE'],
        ]

        final_response = ""
    elif defense_mode_params.name == "query_filter":
        text_to_save = []
        sp_for_filter = read_file(defense_mode_params.path_to_system_prompt_for_filter)
        model_for_filter = defense_mode_params.model_for_filter
        retry_time = 5
        counter = 0
        while counter < retry_time:
            counter += 1
            try:
                filtered_user_query = get_text_generation(
                    model_id=model_for_filter,
                    user_query=user_query,
                    system_prompt=sp_for_filter,
                ).lstrip()
            except openai.BadRequestError as e:  # may be blocked due to jailbreak
                err_msg = f'Query filtering failed due to type{e}: {e}'
                text_to_save += [[
                    'text',
                    err_msg.upper()]
                ]
                continue  # try another time
            break
        if counter >= retry_time:
            logging.info(f"\t\tFailed to filter query "
                         f"after {retry_time} times. "
                         f"Use the original version.")
            filtered_user_query = user_query

        text_to_save += [
            ['text', 'SYSTEM PROMPT:'],
            ['text', system_prompt],
            ['text', 'USER QUERY:'],
            ['text', user_query],
            ['text', 'SYSTEM PROMPT FOR QUERY FILTER:'],
            ['text', sp_for_filter],
            ['text', 'FILTERED QUERY:'],
            ['text', filtered_user_query]
        ]

        final_response = get_text_generation(
            model_id=model,
            user_query=filtered_user_query,
            system_prompt=system_prompt,
            generate_kwargs=generate_kwargs
        )
        text_to_save += [
            ['text', 'FINAL RESPONSE:'],
            ['text', final_response]
        ]
    elif "mll_test_regen" in defense_mode_params.name:
        text_to_save = []
        significance = defense_mode_params.hypothesis_test_significance
        hypothesis_test_threshold, assumed_dist, zero_dist_args, other_dist_args \
            = get_hypothesis_threshold_for_mll_test_regen(
            model_name=model,
            system_prompt=system_prompt,
            defense_mode_params=defense_mode_params,
            significance=significance
        )
        test_info = (f"The threshold {hypothesis_test_threshold} is "
                     f"determined under the significance level {significance}.")

        # because they can be reused across different alpha
        if defense_mode_params.raw_text_dir is not None:
            with_response_path = os.path.join(defense_mode_params.raw_text_dir, 'with.txt')
            if not os.path.isfile(with_response_path):
                first_response = get_text_generation(
                    model_id=model,
                    user_query=user_query,
                    system_prompt=system_prompt,
                    generate_kwargs=generate_kwargs
                )
                with open(with_response_path, 'w') as fout:
                    fout.write(first_response)
            else:
                with open(with_response_path, 'r') as fin:
                    first_response = fin.read()
        else:
            first_response = get_text_generation(
                model_id=model,
                user_query=user_query,
                system_prompt=system_prompt,
                generate_kwargs=generate_kwargs
            )

        likelihood_ratio, likelihood_info = get_likelihood_ratio(
            model_id=model,
            response=first_response,
            system_prompt=system_prompt,
            user_query=user_query,
            dist=assumed_dist,
            zero_dist_args=zero_dist_args,
            other_dist_args=other_dist_args,
            dryrun=False
        )

        # accept null hypothesis, i.e., leakage exists
        if likelihood_ratio >= hypothesis_test_threshold:
            if defense_mode_params.raw_text_dir is not None:
                without_response_path = os.path.join(defense_mode_params.raw_text_dir,
                                                     'without.txt')
                if not os.path.isfile(without_response_path):
                    final_response = get_text_generation(
                        model_id=model,
                        user_query=user_query,
                        generate_kwargs=generate_kwargs
                    )
                    with open(without_response_path, 'w') as fout:
                        fout.write(final_response)
                else:
                    with open(without_response_path, 'r') as fin:
                        final_response = fin.read()

                # if we need to load, we must have regenerated
                # otherwise empty response is no need to load
                indicator = ['text', f'REGENERATING WITHOUT SYSTEM PROMPT AS '
                                     f'{likelihood_ratio} >= {hypothesis_test_threshold}.']
            else:
                if (defense_mode_params.no_regen is not None
                        and defense_mode_params.no_regen is True):
                    final_response = ""
                    indicator = ['text', f'OUTPUT NOTHING AS '
                                         f'{likelihood_ratio} >= {hypothesis_test_threshold}.']
                else:
                    final_response = get_text_generation(
                        model_id=model,
                        user_query=user_query,
                        generate_kwargs=generate_kwargs
                    )
                    indicator = ['text', f'REGENERATING WITHOUT SYSTEM PROMPT AS '
                                         f'{likelihood_ratio} >= {hypothesis_test_threshold}.']
            text_to_save += [
                ['text', 'SYSTEM PROMPT:'],
                ['text', system_prompt],
                ['text', 'USER QUERY:'],
                ['text', user_query],
                ['text', 'FIRST LLM RESPONSE:'],
                ['text', first_response],
                ['text', likelihood_info.upper() + test_info.upper()],
                indicator,
                ['text', 'FINAL RESPONSE:'],
                ['text', final_response],
            ]
        else:
            final_response = first_response
            text_to_save += [
                ['text', 'SYSTEM PROMPT:'],
                ['text', system_prompt],
                ['text', 'USER QUERY:'],
                ['text', user_query],
                ['text', 'FINAL RESPONSE:'],
                ['text', final_response],
                ['text', likelihood_info.upper() + test_info.upper()],
            ]
    elif "cmp_embed_regen" in defense_mode_params.name:
        first_response = get_text_generation(
            model_id=model,
            user_query=user_query,
            system_prompt=system_prompt,
            generate_kwargs=generate_kwargs
        )
        if defense_mode_params.embed_model is not None:
            # this should prevail the argument embed_model
            embed_model = defense_mode_params.embed_model

        first_response_embed = get_text_embedding(
            embed_model_id=embed_model,
            text=first_response
        )
        system_prompt_embed = get_text_embedding(
            embed_model_id=embed_model,
            text=system_prompt
        )
        similarity_score = calc_embed_dist(
            embedding1=first_response_embed,
            embedding2=system_prompt_embed
        )
        similarity_metric = defense_mode_params.similarity_metric
        metric = similarity_score[similarity_metric]
        temp_str = f"[DEBUG] {similarity_score}"  # for ease of debug only
        text_to_save = [
            ['text', 'SYSTEM PROMPT:'],
            ['text', system_prompt],
            ['text', 'USER QUERY:'],
            ['text', user_query]
        ]

        upperbound = defense_mode_params.upperbound
        if metric >= upperbound:
            temp_str_2 = (f"REGENERATING WITHOUT SYSTEM PROMPT "
                          f"AS {similarity_metric.upper()} {metric} >= {upperbound}")
            final_response = get_text_generation(
                model_id=model,
                user_query=user_query,
                generate_kwargs=generate_kwargs
            )
            text_to_save += [
                ['text', 'FIRST LLM RESPONSE:'],
                ['text', first_response],
                ['text', temp_str],
                ['text', temp_str_2],
                ['text', 'FINAL RESPONSE:'],
                ['text', final_response],
            ]
        else:
            final_response = first_response
            text_to_save += [
                ['text', 'FINAL RESPONSE:'],
                ['text', final_response],
                ['text', temp_str]
            ]
    else:
        raise NotImplementedError
    return final_response, text_to_save
