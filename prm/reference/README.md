# 🌐 Demystifying Multilingual Reasoning in Process Reward Modeling


## Overview

Multilingual PRM is a framework for training and evaluating process reward models across multiple languages, focusing on multilingual reasoning. This repository provides code, datasets, and scripts to reproduce the experiments from our paper.

---


## 📦 Datasets


- Multilingual PRM800K: [vicky23456/prm800k-phrase2](https://huggingface.co/datasets/vicky23456/multilingual-PRM800K)
- Multilingual Math Shepherd: [vicky23456/multilingual-mathshepherd](https://huggingface.co/datasets/vicky23456/multilingual-mathshepherd)


Download datasets to the `/data` folder:

```bash
# Example: using huggingface-cli
huggingface-cli download vicky23456/multilingual-PRM800K --local-dir /data
huggingface-cli download vicky23456/multilingual-mathshepherd --local-dir /data
```

---

## 🏋️‍♂️ Training

Train the Multilingual PRM model:

```bash
bash sft.sh
```

---

## 🔍 Analysis & Evaluation

**Sample N candidates:**

```bash
sh infer.sh
```

**Best-of-N Evaluation:**

```bash
sh best-of-n.sh
```

---

## 📊 Results

Our experiments demonstrate that process reward models can effectively generalize reasoning across languages, outperforming standard reward models in multilingual settings. See [our paper](https://arxiv.org/pdf/2502.12663) for detailed results and analysis.

---

## 📖 Paper

If you use this work or datasets, please cite:

```
@article{wang2025demystifying,
  title={Demystifying Multilingual Chain-of-Thought in Process Reward Modeling},
  author={Wang, Weixuan and Wu, Minghao and Haddow, Barry and Birch, Alexandra},
  journal={arXiv preprint arXiv:2502.12663},
  year={2025}
}
```

