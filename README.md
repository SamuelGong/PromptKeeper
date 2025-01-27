# PromptKeeper: Safeguarding System Prompts for LLMs

## 1. Initialization

### 1.1 Necessary Part

Ubuntu assumed.

```bash
git clone git@github.com:SamuelGong/PromptKeeper.git
cd PromptKeeper
conda create -n promptkeeper python=3.9 -y
conda activate promptkeeper
pip install -r requirements.txt --upgrade
```


### 1.2 Installing `output2prompt` (Optional)

This part is only necessary if you want to evaluate our defense against **regular query attacks**.

```bash
# starting from this project directory
cd ..  # or elsewhere you want
git clone git@github.com:SamuelGong/output2prompt.git
cd output2prompt
wget "https://www.dropbox.com/scl/fi/wbun7cj5mdwmd7gzrwv1i/prompt2output_inverters.zip?rlkey=oiyfzhl158nj6zbjqp182mua7&st=2v3wtp2w&dl=0" -O prompt2output_inverters.zip
unzip prompt2output_inverters.zip
cd ../PromptKeeper  # or any way to return back to this project
```

## 2. Toy Example

To verify the installation regarding Step 1.1, try to run the following command:

```bash
python main.py toy_example
```

If everything goes smoothly, there should be a newly generated folder (like the prepared one `20240812-185028`) under `5_iclr_toy` with structure like:

```
|__[some timestamp]
    |__direct
        |__zhang2024_sentences_seen_1.txt
        |__zhang2024_sentences_seen_2.txt
        |__zhang2024_sentences_seen_3.txt
        |__perez2022_ignore_and_spell_check_1.txt
        |__...
    |__mll_test_gen
        |__...
    |__atk_eval_res.pkl
    |__log.txt
```

Here, the `log.txt` records the overall execution process including a high-level summary:

```
Best attack result for defense mode direct: Best cosine similarity: 0.9836376611226733 (zhang2024_sentences_seen 3). Best BLEU: 85.83016467391602 (zhang2024_sentences_seen 3). Best token set F1: 0.9431438127090301 (zhang2024_sentences_seen 3).
Best attack result for defense mode mll_test_regen: Best cosine similarity: 0.7962665013995153 (wallace2024_repeat_verbatim 1). Best BLEU: 0.8627785848958904 (wallace2024_repeat_verbatim 1). Best token set F1: 0.23923444976076552 (zhang2024_sentences_seen 3).
Done in 395.33s.
```

Congratulations if you can see things like that.

**Remarks**
- `direct` means "No defense", `mll_test_regen` means our method.