import os
import sys
import pickle
import google.generativeai as genai

temp_sp_path = 'genai_temp_sp.txt'
temp_uq_path = 'genai_temp_uq.txt'
temp_resp_path = 'genai_temp_resp.txt'
temp_text_path = 'genai_temp_text.txt'
temp_embed_path = 'genai_temp_embed.pkl'



# Main function is for testing only
def main(args):
    genai_api_key = os.getenv('GENAI_API_KEY')
    genai.configure(api_key=genai_api_key)
    method, model = args[:2]

    if method == "generation":
        mode = args[2]
        with open(temp_sp_path, 'r') as fin:
            system_prompt = fin.read()
        with open(temp_uq_path, 'r') as fin:
            user_query = fin.read()

        if mode == "true":  # have system prompt
            model = genai.GenerativeModel(
                model,
                system_instruction=system_prompt
            )
        else:
            model = genai.GenerativeModel(
                model
            )
        generation = model.generate_content(user_query)
        response = generation.text

        with open(temp_resp_path, 'w') as fout:
            fout.write(response)
    elif method == "embedding":
        with open(temp_text_path, 'r') as fin:
            text = fin.read()
        result = genai.embed_content(
            model=model,
            content=text
        )["embedding"]
        with open(temp_embed_path, 'wb') as fout:
            pickle.dump(result, fout)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1:])

