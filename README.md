# 🔬 Spurious Rewards Paradox:  Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs

**[Paper](https://arxiv.org/abs/XXXX)** 

This repository contains the implementation code for our paper:   **"Spurious Rewards Paradox:  Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs"**.  

We provide mechanistic interpretability tools to analyze how spurious reinforcement learning with verifiable rewards (RLVR) activates memorization shortcuts in contaminated models like Qwen2.5-Math-7B. 

---

## Overview

Our work investigates the internal mechanisms through which spurious RLVR training activates memorized solutions in LLMs. We identify:  

- **The Perplexity Paradox**: Answer perplexity decreases while prompt perplexity increases during spurious RLVR
- **Functional Anchor (L18-L20)**: Middle layers that inject the decisive memorization trigger
- **Structural Adapters (L21+)**: Later layers that transform representations to accommodate shortcuts
- **Causal Steering**:  Neuron-level interventions to control memorization behavior

---

##  Setup

```bash
# Our codebase is based on TTRL (https://github.com/PRIME-RL/TTRL) and
# Spurious_Rewards(https://github.com/ruixin31/Spurious_Rewards)
# You can start by cloning the repo:
git clone git@github.com:ruixin31/Spurious_Rewards
cd code

# Create conda environment and install our dependencies
conda create -n rlvr-mechanism python=3.10
conda activate rlvr-mechanism

pip install -r requirements.txt
pip install flash_attn==2.7.0.post2
```

---

##  Training & Data

For RLVR training and data preparation, please refer to the base repository:

**[Spurious Rewards](https://github.com/ruixin31/Spurious_Rewards)**

Our analysis is built on top of models trained using their framework with different reward signals (ground-truth, random, incorrect, format-only).

**Note**: We have additionally trained OLMo, LLaMA, and Qwen3 models using this framework. **These model checkpoints will be released soon**.


---

## Main Analysis Scripts

Our repository provides the following key analysis tools:

- **Perplexity Analysis**: Identify the Perplexity Paradox phenomenon
- **Path Patching**:  Causally localize functional layers
- **JSD Analysis**: Quantify structural changes in MLP components
- **Logit Lens**: Visualize token emergence across layers
- **Neural Differential Equations**: Model trajectory bifurcation
- **Ablation Studies**: Test necessity and sufficiency of identified circuits
- **Neuron Steering**: Causally intervene on memorization behavior

See `scripts/` for detailed usage examples.

---




## Citation

If you find our work helpful, please cite: 

```bibtex
placeholder
```


---

##  Acknowledgments

This repository builds upon the excellent work of: 

- **[Spurious Rewards](https://github.com/ruixin31/Spurious_Rewards)** for the RLVR training framework and initial observations
- **[TTRL](https://github.com/PRIME-RL/TTRL)** for the RL training infrastructure
- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** for the foundational RLHF toolkit
- **[TransformerLens](https://github.com/neelnanda-io/TransformerLens)** for interpretability utilities

We thank the authors for open-sourcing their code and making reproducible research possible.


