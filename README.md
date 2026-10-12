# Retrieval-Augmented Large Language Models Enable Training-Free Kinase–Substrate Phosphorylation Prediction from Protein Databases and Biomedical Literature

![Framework](figures/fig1_architecture.png)

## Overview

For each kinase–substrate pair, we retrieve UniProt annotations and the top-3 PubMed abstracts, then ask an LLM "Yes" or "No". No training is needed.

## Run

```bash
pip install -r requirements.txt
python baselines/baselines.py
python scripts/make_figures.py
```

## Citation

```bibtex
@inproceedings{azam2026ragkinase,
  title     = {Retrieval-Augmented Large Language Models Enable Training-Free Kinase--Substrate
               Phosphorylation Prediction from Protein Databases and Biomedical Literature},
  author    = {Azam, Muhammad and ABC and Zhao, Zhongming},
  booktitle = {Proc. IEEE BIBM},
  year      = {2026}
}
```
