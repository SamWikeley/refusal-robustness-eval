# AI Safety: Refusal Robustness Evaluation

A systematic evaluation framework for testing the robustness of large language models against jailbreak attempts via prompt variation.


## Overview

This project implements a comprehensive evaluation framework deseigned to assess the propensity of large language models (LLMs) to comply with adversarial prompts. The framework is implemented from scratch and supports both custom datasets and industry-standard benchmarks, such as AdvBench [[1]](#1). 

Given a dataset of intentionally malicious requests, the program automatically generates an evaluation dataset based on 7 distinct prompt variations designed to bypass LLM safety protocols. Each evaluation prompt is sent to a target model via API integration and the model responses are classified by a refusal detection algorithm. The program computes refusal rates by attack type and identifies the most effective jailbreak methods.  


## Key Features

- **Multi-variant jailbreak testing:** 7 distinct prompt variations (direct, polite, artistic, roleplay, hypothetical, educational, authority)
- **Automated refusal detection:** NLP-based system to clasify model responses 
- **Statistical analysis:** Comprehensive reporting of refusal rates and potential vulnerabilities
- **AdvBench integration:** Support for industry-standard AI safety benchmarks
- **Scalable pipeline:** Simple extension to further models and datasets, and more sophisticated evaluation methodologies


## Methodology

The evaluation framework implements the following workflow:

1. **Dataset creation:** Generate systematic variations of harmful requests from either user-defined or established datasets
2. **Model testing:** Send prompts to target models via API with rate limiting and error handling
3. **Response analysis:** Automated classification of model responses as "compliance" or "refusal"
4. **Statistical analysis:**  Calculate refusal rates by attack type and identify most effective jailbreak variations 

### Jailbreak Variants:

- **Direct:** Original harmful request without modification
- **Polite:** Add politeness to request
- **Artistic:** Frame request as needed for fictional scenario
- **Roleplay:** Ask model to take on a role
- **Hypothetical:** Present request as a theoretical discussion
- **Educational:** Claim request is for educational or research purposes
- **Authority:** Claim authority and that request is for professional need


## Set-Up Guide

1. Clone the repository:
```bash
git clone https://github.com/samwikeley/refusal-robustness-eval.git
cd refusal-robustness-eval
```

2. Create environment file:
```bash
cp .env.example .env
# Add your API keys to .env
```

3. Generate evaluation dataset:
```bash
python src/custom_setup.py
python src/advbench_setup.py --subset {int} --test {int}
# Arguments: 
# 'subset' - subset size of full AdvBench dataset (default: None) 
# 'test' - size of small dataset for testing (default: 20)
```

4. Run evaluation:
```bash
python src/model_evaluation.py --dataset {str} --model {str} --max {int}
# Arguments: 
# 'dataset' - dataset to evaluate (default: custom)
# 'model' - model to test (default: gpt-3.5-turbo) 
# 'max' - maximum number of test cases to run (default: None)
```


## Disclaimer

This tool is designed for AI safety research purposes only. Valid use-cases are:

- AI evaluation research

- Model robustness testing

- Educational purposes in AI safety 


## References

<a id="1">[1]</a> 
Zou et al (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models [2307.15043](https://arxiv.org/abs/2307.15043)
