# AutoRec++
This is the implementation for the paper work of "A Debiasing Autoencoder for Recommender System"

Author: Teng Huang, Cheng Liang, Di Wu, and Yi He

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
## Brief Introduction
In this paper, we aim at comprehensively addressing the various biases existed in user behavior data for DNN-based RSs. To this end, we incorporate various combinations of preprocessing bias (PB) and training bias (TB) into the Autoencoder to obtain the optimal bias combination. After obtaining the optimal bias combination we further train the model under L1-L2-norm to propose our AutoRec++ model. By conducting extensive experiments on five benchmark datasets, we demonstrate that: 1) the Autoencoder’s prediction accuracy and computational efficiency can be significantly boosted by incorporating the optimal combination of PB and TB into it without structural change, and 2) our AutoRec++ achieves significantly better prediction accuracy and robustness to outliers than both DNN-based and non-DNN-based state-of-the-art models.

## Files

The overall framework of this project is designed as follows

1. The **data** folder is used to hold the datasets;
2. The **models** folder is used to store the proposed model;
3. The **utils** folder saves tool scripts including evaluation method, loss function, and training results saver, etc.;
4. The **results** folder saves the result of the training process and the parameters of AutoRec++;

### Enviroment Requirement
- numpy
- tensorflow (below 2.0 otherwise need to call disable_v2_behavior())

### Dataset
We offer all the dataset with three different train-test ratio involved in the experiment.


### Getting Started
1. Clone this repository

```angular2html
git clone https://github.com/wudi1989/AutoRec_Plus.git
```

2. Make sure you meet package requirements by running:

```angular2html
pip install -r requirement.txt
```
3. Train and test AutoRec++ model

```angular2html
python main.py
```

#### Improtant arguments in main.py
- `data_name`: to choose different datasets;
  - options: "Ml1M", "Ml100k", "Hetrec-ML", "Yahoo", "douban"
- `isL1L2`: boolean, to choose whether train AutoRec++ under L1-L2-norm loss function;
  - Ture: train AutoRec++ under L1-L2-norm loss function
  - False: train AutoRec++ under L2-norm loss function

- _other hyperparameters are fine-tuned and shown in main.py_

``
For example, to train and test AutoRec++ on Ml1M under L1-L2-norm loss funtion, just run:
``
```angular2html
python main.py --data_name "Ml1M" --isL1L2 Ture
```
