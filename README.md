# [NeurIPS 2026] SLDR: Defending Against Malicious Fine-tuning via Selective Layers Recovery and Dynamic Routing

## 🔧 Environment Setup

Clone the repository, create and activate the Conda environment, then install the required dependencies:

```bash
git clone https://github.com/Stardust457/SLDR.git
cd SLDR

conda create -n sldr python=3.12 -y

conda activate sldr

python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

---

## 🔑 Accessing and Downloading Llama Models

Llama models require an access request on Hugging Face.

Before using a model, log in to [Hugging Face](https://huggingface.co/), visit the desired model's page under [Meta Llama](https://huggingface.co/meta-llama), accept the terms of use, and submit an access request. Once access is granted, use the same account to create an Access Token with read access to the target model on the [Access Tokens settings page](https://huggingface.co/settings/tokens). For details, see the [Hugging Face gated model guide](https://huggingface.co/docs/hub/models-gated).

First, configure the [HF-Mirror](https://hf-mirror.com/) in a Bash terminal:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Then, run the following command to log in:

```bash
hf auth login
```

Enter your newly created Hugging Face Access Token when prompted. 

---

## 🚀 Training and Evaluation

Run the following commands from the repository root to train and evaluate each model:

### Llama 3.1

```bash
bash scripts/run_llama31_downstream.sh
```

### Llama 3

```bash
bash scripts/run_llama3_downstream.sh
```

### Qwen 2.5

```bash
bash scripts/run_qwen25_downstream.sh
```

### Mistral

```bash
bash scripts/run_mistral_downstream.sh
```

### Llama Guard Deployment and Harmfulness Scoring

After collecting model responses, deploy Llama Guard and start the service in a terminal:

```bash
bash scripts/run_llama_guard.sh serve
```

Keep this terminal running and wait until the service is ready. Then, open another terminal, activate the `sldr` environment, and run the following command from the repository root to score the harmfulness of the model responses:

```bash
bash scripts/run_llama_guard.sh score
```

### ⚠️ Important Reminder: Alpaca Evaluation

Before running Llama 3.1 evaluation on the Alpaca dataset, configure the GPT judge's API base URL and API key in the same terminal:

```bash
export GPT_JUDGE_BASE_URL='your base url'
export OPENAI_API_KEY='your API key'
```

Replace the placeholders with your API base URL and API key before starting the evaluation.

### Additional Harmful Benchmark Evaluation

After generating the corresponding downstream checkpoints, run the following command to evaluate Llama 3.1 on additional harmful benchmarks, including DirectHarm4, HarmBench, and HEx-PHI:

```bash
DATASETS="sst2" \
POISON_RATIOS="0.1" \
SEEDS="42" \
VARIANTS="defended" \
bash /root/scripts/run_llama31_harm_benchmarks.sh
```

### Zulu and IJP Jailbreak Evaluation

Run the following command to evaluate Llama 3.1 against the Zulu and IJP jailbreak attacks using the corresponding downstream checkpoints:

```bash
DATASETS="sst2" \
POISON_RATIOS="0.1" \
TRAIN_SAMPLES="1000" \
SEEDS="42" \
VARIANTS="defended" \
JAILBREAK_ATTACKS="zulu ijp" \
bash /root/scripts/run_llama31_jailbreak.sh
```

**Note:** Model responses generated for the Zulu dataset must be translated into English before their harmfulness is evaluated using Llama Guard.

---
