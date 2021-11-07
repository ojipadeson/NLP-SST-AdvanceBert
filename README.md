# SST-2-sentiment-analysis

Use BERT, RoBERTa, XLNet and ALBERT models to classify the SST-2 data set based on pytorch.

Codes are runned on Nvidia Tesla K80(2496x2 cuda core, 12x2GB RAM)

### Result on test
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 87.6 | 87.8 | 87.8 | 87.6 | 110M |
RoBERTa	| **89.2**	| **89.2**	| **89.2**	| **89.2** | 223M |
XLNet	|  |  |  |  | 125M |
ALBERT	| 87.6 | 87.6 | 87.6 | 87.6 | 340M |

* *Albert is really hard to train*

### Result on dev
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 87.2 | 87.4 | 87.2 | 87.2 | 110M |
RoBERTa	| **89.1**	| **89.1**	| **89.1**	| **89.1** | 223M |
XLNet	|  |  |  |  | 125M |
ALBERT	| 86.7 | 86.7 | 86.7 | 86.7 | 340M |

### Result on test -- On PJ dataset
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 86.9  | 87.1  | 87.0	| 86.9 | 110M |
RoBERTa	|  |  |  |  | 223M |
XLNet	|  |  |  |  | 125M |
ALBERT	|  |  |  |  | 340M |

### Result on dev -- On PJ dataset
 Model | Accuracy | Precision	| Recall | F1 | Parameters |
 ----   | -----  |----- |----- |----- |----- 
 BERT   | 87.2  | 87.3  | 87.2	| 87.2 | 110M |
RoBERTa	|  |  |  |  | 223M |
XLNet	|  |  |  |  | 125M |
ALBERT	|  |  |  |  | 340M |

* bert-base-uncased: 12-layer, 768-hidden, 12-heads, trained on lower-cased English text.
* albert-xxlarge-v2: 12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 
  ALBERT xxlarge model with no dropout, additional training data and longer training
  
* roberta-base: 12-layer, 768-hidden, 12-heads, RoBERTa using the BERT-base architecture
* xlnet-large-cased: 24-layer, 1024-hidden, 16-heads, XLNet Large English model

https://huggingface.co/transformers/pretrained_models.html

## Test on Windows
You may encounter OSerror for pytorch < 1.3, because pth file larger than 2GB.
So it's recommended to test Bert & Roberta model for the first step 

## LICENSE
### Reimplement from **github@YJiangcm**, welcome star
* With additional datasets
* Easier use on Lab
* More info for training
* Perfectly suitable for NLP class project

