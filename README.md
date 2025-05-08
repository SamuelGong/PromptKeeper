# PromptKeeper: Safeguarding System Prompts for LLMs

> You can also explore the [DeepWiki](https://deepwiki.com/SamuelGong/PromptKeeper) for this repository, which offers additional insights and helps with answering questions.

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
