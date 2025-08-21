# PromptKeeper: Safeguarding System Prompts for LLMs

> You can also explore the [DeepWiki](https://deepwiki.com/SamuelGong/PromptKeeper) for this repository, which offers additional insights and helps with answering questions.

This repository contains the evaluation artifacts of our paper titled 
*PromptKeeper: Safeguarding System Prompts for LLMs*, 
which has been accepted to the Findings of EMNLP'25.
You can find the paper [here](https://openreview.net/forum?id=1h4RHO6xoN).

[Zhifeng Jiang](https://samuelgong.github.io/), [Zhihua Jin](https://jnzhihuoo1.github.io/), [Guoliang He](https://www.cst.cam.ac.uk/people/gh512/)

**Keywords**: LLM, System Prompt, Security and Privacy

<details> <summary><b>Abstract (Tab here to expand)</b></summary>

System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

</details>

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

## 3. Support
If you need any help, please submit a Github issue, or contact Zhifeng Jiang via zjiangaj@connect.ust.hk.

## 4. Citation

If you find this repository useful, please consider giving ⭐ and citing our paper:

```bibtex
@inproceedings{jiang2025safeguarding,
  author={Jiang, Zhifeng and Jin, Zhihua and He, Guoliang},
  title={Dordis: Efficient Federated Learning with Dropout-Resilient Differential Privacy},
  year={2025},
  booktitle={Findings of EMNLP},
}
```
