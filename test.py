import os
from sys import platform

import torch
import pandas as pd
from torch.utils.data import DataLoader

from data_sst2 import DataPrecessForSentence
from models import BertModel, AlbertModel, RobertModel, XlnetModel
from utils import test

test_df = pd.read_csv(os.path.join('data/', "test_pj.tsv"), sep='\t', header=None, names=['similarity', 's1'])
target_dir = "output/Bert/"
max_seq_len = 50
batch_size = 32

bertmodel = BertModel(requires_grad=False)
# bertmodel = AlbertModel(requires_grad=False)
# bertmodel = RobertModel(requires_grad=False)
# bertmodel = XlnetModel(requires_grad=False)
tokenizer = bertmodel.tokenizer
device = torch.device("cuda")

print(20 * "=", " Preparing for testing ", 20 * "=")
if platform == "linux" or platform == "linux2":
    checkpoint = torch.load(os.path.join(target_dir, "best.pth"))
else:
    checkpoint = torch.load(os.path.join(target_dir, "best.pth"), map_location=device)

print("\t* Loading test data...")
test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=max_seq_len)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Retrieving model parameters from checkpoint.
print("\t* Building model...")
model = bertmodel.to(device)
model.load_state_dict(checkpoint["model"])
print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")

batch_time, total_time, accuracy, all_prob = test(model, test_loader)
print(
    "\n-> Average batch processing time:"
    " {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy * 100)))

test_prediction = pd.DataFrame({'prob_1': all_prob})
test_prediction['prob_0'] = 1 - test_prediction['prob_1']
test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
test_prediction = test_prediction[['prediction']]

test_prediction.to_csv('prediction.tsv', sep='\t')
