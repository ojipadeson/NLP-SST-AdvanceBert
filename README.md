# SST-2-sentiment-analysis Using Bert

---

CopyRight Notice: Modified from **github: @YJiangcm / SST-2-sentiment-analysis**

---

Since Word2Vec, GloVe, etc word embedding can only get <80% accuracy on the pj dataset, this repo
use BERT, RoBERTa, XLNet and ALBERT models to classify the SST-2 data set based on pytorch.
(You can find Word2Vec, GloVe implementation on my other repos)

Codes are runned on Nvidia Tesla K80(2496x2 cuda core, 12x2GB RAM)

In this repo, a wider range of sentences is added to the dataset, which makes the task harder

---

The "pj dataset" is generated from Stanford Sentiment Treebank,
and divided to binary set according to sentiment label(float number range from 0-1)

The classifying boundary is 0.5

---

Above rules are not sure, for details, the pj is conducted by *TA:* **github: @txsun1997**

---

### Result on test
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 87.6 | 87.8 | 87.8 | 87.6 | 110M |
RoBERTa	| 89.2 | 89.2 | 89.2 | 89.2 | 223M |
**XLNet** | **90.2** | **90.2** | **90.3** | **90.2** | 125M |
ALBERT	| 87.6 | 87.6 | 87.6 | 87.6 | 340M |

* *Albert is really hard to train*
* *Roberta & Xlnet is more train-friendly*

### Result on dev
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 87.2 | 87.4 | 87.2 | 87.2 | 110M |
RoBERTa	| 89.1 | 89.1 | 89.1 | 89.1 | 223M |
**XLNet** | **89.6** | **89.6** | **89.6** | **89.6** | 125M |
ALBERT	| 86.7 | 86.7 | 86.7 | 86.7 | 340M |

---

### Result on test -- On PJ dataset
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 86.9 | 87.1 | 87.0 | 86.9 | 110M |
RoBERTa	| 89.5 | 89.5 | 89.4 | 89.4 | 223M |
**XLNet**	| **90.4** | **90.4** | **90.4** | **90.4** | 125M |
ALBERT	|  |  |  |  | 340M |

### Result on dev -- On PJ dataset
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 87.2  | 87.3  | 87.2	| 87.2 | 110M |
RoBERTa	| 88.9 | 88.9 | 88.9 | 88.9 | 223M |
**XLNet**	| **90.6** | **90.6** | **90.6** | **90.6** | 125M |
ALBERT	|  |  |  |  | 340M |

---

* bert-base-uncased: 12-layer, 768-hidden, 12-heads, trained on lower-cased English text.
* albert-xxlarge-v2: 12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 
  ALBERT xxlarge model with no dropout, additional training data and longer training
  
* roberta-base: 12-layer, 768-hidden, 12-heads, RoBERTa using the BERT-base architecture
* xlnet-large-cased: 24-layer, 1024-hidden, 16-heads, XLNet Large English model

https://huggingface.co/transformers/pretrained_models.html

---

## Run
```
python run_Bert_model.py -s -p
```
```-s```  to save the best model to .pth

```-p```  to use the pj train data

Delete them if you don't need them

Obviously you can change ```run_Bert_model.py``` to any similar file in this project.

The **accuracy, score metrics** will be shown on ```logs(stdout)```, and ```prediction.tsv``` will be saved in ```./output```

**If you want to simply get all models run on a GPU server using jupyter notebook, simply do:**
1. Fork the repo to your Github
2. Change the first line of RUN_ALL.ipynb to your Github username and token
3. Click ```>run all``` and wait

## Test on Windows
```
python test.py --albert
```
Obviously you can change ```albert``` to any similar model stored in ```./output```

You may encounter OSerror for pytorch < 1.4, because .pth file is larger than 2GB.
If that happened it's recommended to test Bert & Roberta model for the first step 

---

## Tips
* It seems that dev loss is not a suitable indicator to decide training process, 
such as learning rate, early stopping and so on.
  
* The approximate size of each model.pth: Bert: 1.3GB; Albert: 2.4GB; Roberta: 1.5GB; Xlnet: 4.2GB. 
Be careful with the disk space.
  
* On device K80, the average training time: Albert: 1.5h(20min/epoch); Other: 1h(4min/epoch).
You can change early stopping or epoch parameter according to this.

## LICENSE
### It's a repo for NLP class project
### Reimplementation of some SOTA models
### With several more features:
* With additional datasets
* Easier use on Lab
* More info for training
* Perfectly suitable for NLP class project(if you find your accuracy unsatisfied)

---

## Description

```
* Training epoch 10:
Avg. batch proc. time: 0.6357s, loss: 0.0235: 100%|█| 267/267 [02:50<00:00,  1.5
-> Training time: 170.3649s, loss = 0.0235, accuracy: 99.4148%
* Validation for epoch 10:
-> Valid. time: 6.7613s, loss: 0.5022, accuracy: 90.5540%, auc: 0.9457

Accuracy: 90.6%
Precision: 90.6%
Recall: 90.6%
F1: 90.6%
classification_report:

              precision    recall  f1-score   support

     class_0      0.925     0.885     0.905       558
     class_1      0.887     0.926     0.906       543

    accuracy                          0.906      1101
   macro avg      0.906     0.906     0.906      1101
weighted avg      0.906     0.906     0.906      1101
```

---
