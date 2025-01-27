import os
import json
import logging
import fnmatch
from datetime import datetime


def set_log(log_path):
    log_level = "info"  # TODO: avoid hard-coding
    log_level = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARN,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }[log_level]

    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='[%(levelname)s][%(asctime)s.%(msecs)03d] '
               '[%(filename)s:%(lineno)d]: %(message)s',
        level=log_level,
        datefmt='(%Y-%m-%d) %H:%M:%S'
    )
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def list_txt_files(directory):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


def read_file(file_path):
    with open(file_path, "r") as fin:
        text = fin.read()
    return text


def write_file(file_path, text):
    with open(file_path, "w") as fout:
        fout.write(text)


def save_customized_log(data_list, file_prefix=None, config_to_log=None,
                        need_postfix=True, only_return_text=False):
    if file_prefix is not None:
        if need_postfix:
            file_path = os.path.join(file_prefix,
                                     datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".txt")
        else:
            file_path = file_prefix

    text = ""
    if config_to_log is not None:
        text = f"CONFIG:\n{json.dumps(config_to_log, indent=4)}"

    for idx, eval_res in enumerate(data_list):
        t, item = eval_res
        if t == "json":
            text += '\n\n' + json.dumps(item, indent=4)
        elif t == "text":
            # print(t, item)
            text += '\n\n' + item
        else:
            raise NotImplementedError

    if only_return_text:
        return text
    else:
        write_file(file_path, text)


def find_file_with_pattern(folder_path, pattern):
    # Ensure the pattern ends with a wildcard to match any file that starts with the pattern
    pattern_with_wildcard = pattern + '*'

    # Walk through the folder and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern_with_wildcard):
                return os.path.join(root, filename)
    return None
