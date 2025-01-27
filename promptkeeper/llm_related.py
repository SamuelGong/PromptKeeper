import os
import gc
import time
import torch
import pickle
import logging
import subprocess
import transformers
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from promptkeeper.genai_temp import (temp_uq_path, temp_sp_path,
                                temp_resp_path, temp_text_path, temp_embed_path)

logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# for saving memory space
_transformers_cached_model_id = ""
_transformers_cached_model = None
_transformers_cached_pipeline = None

_transformers_cached_embed_model_id = ""
_transformers_cached_embed_model = None

_google_api_key_set = False
_open_ai_cached_client = None


# Reference: https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.apply_chat_template
# https://huggingface.co/docs/transformers/main/en/chat_templating
def get_prompt(tokenizer, user_query, system_prompt=None):
    if system_prompt is None:
        conversation = []
    else:
        conversation = [
            {'role': 'system', 'content': system_prompt}
        ]

    if not isinstance(user_query, list):
        user_query = [user_query]
    for idx, item in enumerate(user_query):
        if idx % 2 == 0:
            conversation.append({'role': 'user', 'content': item})
        else:
            conversation.append({'role': 'assistant', 'content': item})

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    return prompt


def get_openai_client():
    global _open_ai_cached_client
    if _open_ai_cached_client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError(
                "Environment variable 'OPENAI_API_KEY' is not set. Please set it before running the code.")
        base_url = os.getenv("BASE_URL")
        if not base_url:
            raise EnvironmentError(
                "Environment variable 'BASE_URL' is not set. Please set it before running the code.")

        _open_ai_cached_client = OpenAI(
            api_key=openai_api_key,
            base_url=base_url,
        )
    return _open_ai_cached_client


def set_google_ai_key():
    global _google_api_key_set
    if _google_api_key_set is False:
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Environment variable 'GENAI_API_KEY' is not set. Please set it before running the code.")
        genai.configure(api_key=api_key)


def get_text_embedding(embed_model_id, text):
    if embed_model_id in ["text-embedding-ada-002"]:
        client = get_openai_client()
        retry_time = 10
        retry_iter = 1
        while True:
            try:
                result = client.embeddings.create(
                    model=embed_model_id,
                    input=text,
                    encoding_format="float"
                ).data[0].embedding
            except Exception as e:
                if retry_iter >= retry_time:
                    raise e

                retry_iter += 1
                print(f'Encountered exception: {e} when '
                      f'getting text-embedding with {embed_model_id}, '
                      f'will retry the {retry_iter}-th time...')
                time.sleep(1)
            else:
                break

    elif embed_model_id in ["models/text-embedding-004"]:
        # ref: https://ai.google.dev/api/python/google/generativeai/embed_content
        # result = genai.embed_content(
        #     model=config.embed_model,
        #     content=text
        # )["embedding"]

        # see test_gemini.py to know why the above shortcut is not used
        # but the following weird code
        set_google_ai_key()
        command = ['python', 'genai_temp.py', 'embedding', f"{embed_model_id}"]
        with open(temp_text_path, 'w') as fout:
            fout.write(text)
        subprocess.run(command)
        with open(temp_embed_path, 'rb') as fin:
            result = pickle.load(fin)
    elif embed_model_id in [
        "all-mpnet-base-v2",
        "average_word_embeddings_komninos"
    ]:
        # otherwise the model has to be loaded for every function call,
        # and/or the memory will be run out if using too many models of different kinds
        global _transformers_cached_embed_model_id, \
            _transformers_cached_embed_model
        if not embed_model_id == _transformers_cached_embed_model_id:
            if _transformers_cached_embed_model is not None:
                del _transformers_cached_embed_model
                gc.collect()
                torch.cuda.empty_cache()

            _transformers_cached_embed_model_id = embed_model_id
            embed_model = SentenceTransformer(embed_model_id)
            _transformers_cached_embed_model = embed_model
        else:
            embed_model = _transformers_cached_embed_model
        result = embed_model.encode(text)  # class 'numpy.ndarray' with 768
    else:
        raise NotImplementedError

    return result


def get_text_generation(model_id, user_query, system_prompt=None,
                        generate_kwargs=None):
    if model_id in ["gpt-35-turbo", "gpt-4", "gpt-4o"]:
        if isinstance(user_query, list):
            raise NotImplementedError

        if system_prompt is not None:
            message_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        else:
            message_list = [
                {"role": "user", "content": user_query}
            ]

        client = get_openai_client()
        if generate_kwargs is None:
            generate_kwargs = {}
        # retry_time = 10
        # retry_iter = 1
        # while True:
        #     try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=message_list,
            **generate_kwargs
        )

        response = completion.choices[0].message.content
    elif model_id in ["gemini-1.5-pro"]:
        if isinstance(user_query, list):
            raise NotImplementedError

        set_google_ai_key()
        command = ['python', 'genai_temp.py', 'generation', f"{model_id}"]
        with open(temp_uq_path, 'w') as fout:
            fout.write(user_query)

        if system_prompt is not None:
            with open(temp_sp_path, 'w') as fout:
                fout.write(system_prompt)
            command.append('true')
        else:
            command.append('false')
        subprocess.run(command)

        with open(temp_resp_path, 'r') as fin:
            response = fin.read()
    elif model_id in ["meta-llama/Llama-2-7b-chat-hf",
                      "meta-llama/Llama-2-7b-hf",
                      "meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "mistralai/Mistral-7B-Instruct-v0.3"]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs = {
            "device_map": "auto",
            # "max_memory": {0: "16GB"}  # TODO: for AWS EC2 p3.2xlarge only
        }

        if model_id in ["meta-llama/Meta-Llama-3.1-8B-Instruct"]:
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                total_mem_in_gigabytes = total_mem / 1000000000
                if total_mem_in_gigabytes < 18:  # TODO: avoid hard-coding
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )  # 8-bit
                    model_kwargs["quantization_config"] = quantization_config
                    # print('[Debug] 8bit for Llama')
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                    # print('[Debug] 16bit for Llama')
        else:  # otherwise 16-bit
            model_kwargs["torch_dtype"] = torch.float16

        # otherwise the model has to be loaded for every function call,
        # and/or the memory will be run out if using too many models of different kinds
        global _transformers_cached_model_id, \
            _transformers_cached_model, \
            _transformers_cached_pipeline
        if not model_id == _transformers_cached_model_id:
            if _transformers_cached_model is not None:
                del _transformers_cached_model
                if _transformers_cached_pipeline is not None:
                    del _transformers_cached_pipeline
                # only these two "del"'s combined will work
                gc.collect()
                torch.cuda.empty_cache()

            _transformers_cached_model_id = model_id
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            _transformers_cached_model = model
        else:
            model = _transformers_cached_model

        _transformers_inference_pipeline \
            = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        _transformers_cached_pipeline = _transformers_inference_pipeline

        prompt = get_prompt(tokenizer, user_query, system_prompt)
        if generate_kwargs is None:
            generate_kwargs = {}
        if model_id in [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ]:
            generate_kwargs["max_new_tokens"] = 1024  # important for Mistral models
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
        len_prompt = len(prompt)

        sequences = _transformers_inference_pipeline(
            prompt,
            do_sample=True,
            **generate_kwargs
        )
        response = sequences[0].get("generated_text")[len_prompt:]

        torch.cuda.empty_cache()
    else:
        raise NotImplementedError(f"Model {model_id}")

    return response
