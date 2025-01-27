import os
import pickle
import sys
import json
import time
import logging
from datetime import datetime
from promptkeeper.config import load_configurations
from promptkeeper.utils import (read_file, set_log)
from promptkeeper.evaluate_attack import fast_analysis
from promptkeeper.defense import (get_upperbound_for_cmp_embed_regen,
                     before_get_upperbound_for_cmp_embed_regen)
from promptkeeper.attack import (adversarial_query_attack, regular_query_attack)


def main(args):
    overall_start_time = time.perf_counter()
    exp_path = os.path.join(os.getcwd(), args[0])  # os.getcwd() is important
    config = load_configurations(exp_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    execution_log_path = os.path.join(exp_path, f'{timestamp}_log.txt')
    set_log(execution_log_path)
    logging.info(f"CONFIG:\n{json.dumps(config, indent=4)}")
    print(f"One may see the log in real-time via command: less +F {execution_log_path}")

    raw_res_for_save_path = os.path.join(exp_path, 'atk_eval_res.pkl')
    if os.path.isfile(raw_res_for_save_path):
        with open(raw_res_for_save_path, 'rb') as fin:
            raw_res_for_save = pickle.load(fin)
        logging.info(f"Previous results loaded from {raw_res_for_save_path}")
    else:
        raw_res_for_save = {}

    for defense_idx, defense_mode_params in enumerate(config.defense_mode):
        defense_mode = defense_mode_params.name
        if defense_mode not in raw_res_for_save:
            raw_res_for_save[defense_mode] = {}
        mode_dir = os.path.join(exp_path, defense_mode)
        os.makedirs(mode_dir, exist_ok=True)
        logging.info(f"Testing defense: {defense_mode} "
                     f"({defense_idx + 1}/{len(config.defense_mode)}).")

        if (defense_mode == "cmp_embed_regen" and
                defense_mode_params.upperbound_compute_params is not None):
            defense_mode_params, embed_model \
                = before_get_upperbound_for_cmp_embed_regen(config, defense_mode_params)
            upperbound = get_upperbound_for_cmp_embed_regen(
                model=config.model,
                embed_model=embed_model,
                upperbound_compute_params=
                defense_mode_params.upperbound_compute_params
            )
            defense_mode_params.upperbound = upperbound

        system_prompt_filenames = [file for file in os.listdir(config.system_prompt_dir)
                                   if file.endswith('.txt')]
        for prompt_id, system_prompt_filename in enumerate(system_prompt_filenames):
            system_prompt_name = system_prompt_filename.split('/')[-1].split('.')[0]
            system_prompt_path = os.path.join(config.system_prompt_dir, system_prompt_filename)
            system_prompt = read_file(system_prompt_path)
            if system_prompt_name not in raw_res_for_save[defense_mode]:
                raw_res_for_save[defense_mode][system_prompt_name] = {}
            prompt_dir = os.path.join(mode_dir, system_prompt_name)
            os.makedirs(prompt_dir, exist_ok=True)
            logging.info(f"\tUsing system prompt {system_prompt_name} "
                         f"({prompt_id + 1}/{len(system_prompt_filenames)}).")

            if config.user_query_dir is not None:
                adversarial_query_attack(
                    config=config,
                    defense_mode_params=defense_mode_params,
                    system_prompt=system_prompt,
                    system_prompt_name=system_prompt_name,
                    prompt_dir=prompt_dir,
                    raw_res_for_save_path=raw_res_for_save_path,
                    raw_res_for_save=raw_res_for_save
                )
            else:
                regular_query_attack(
                    config=config,
                    defense_mode_params=defense_mode_params,
                    system_prompt=system_prompt,
                    system_prompt_name=system_prompt_name,
                    prompt_dir=prompt_dir,
                    raw_res_for_save_path=raw_res_for_save_path,
                    raw_res_for_save=raw_res_for_save
                )

        _, outmost_res_for_print = fast_analysis(raw_res_for_save[defense_mode])
        logging.info(f"Best mean attack result for defense mode {defense_mode}: {outmost_res_for_print}")
        _, outmost_res_for_print_mean = fast_analysis(raw_res_for_save[defense_mode], mode="mean")
        logging.info(f"Mean mean attack result for defense mode {defense_mode}: {outmost_res_for_print_mean}")

    overall_end_time = time.perf_counter()
    overall_duration = overall_end_time - overall_start_time
    logging.info(f"Done in {round(overall_duration, 2)}s.")


if __name__ == "__main__":
    main(sys.argv[1:])
