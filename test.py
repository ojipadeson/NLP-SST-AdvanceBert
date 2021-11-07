import os
import argparse
from sys import platform

import torch
import pandas as pd
from torch.utils.data import DataLoader

from data_sst2 import DataPrecessForSentence
from models import BertModel, AlbertModel, RobertModel, XlnetModel
from utils import test, Metric

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--albert', action='store_true',
                    help='run albert model')
parser.add_argument('--albert_pj', action='store_true',
                    help='run albert model on pj train dataset')
parser.add_argument('--roberta', action='store_true',
                    help='run albert model')
parser.add_argument('--roberta_pj', action='store_true',
                    help='run roberta model on pj train dataset')
parser.add_argument('--xlnet', action='store_true',
                    help='run xlnet model')
parser.add_argument('--xlnet_pj', action='store_true',
                    help='run xlnet model on pj train dataset')
parser.add_argument('--dev', action='store_true',
                    help='run test on pj dev dataset')
args = parser.parse_args()

if args.albert:
    target_dir = 'output/Albert/'
    bertmodel = AlbertModel(requires_grad=False)
elif args.albert_pj:
    target_dir = 'output/Albert-pj/'
    bertmodel = AlbertModel(requires_grad=False)
elif args.roberta:
    target_dir = 'output/Roberta/'
    bertmodel = RobertModel(requires_grad=False)
elif args.roberta_pj:
    target_dir = 'output/Roberta-pj/'
    bertmodel = RobertModel(requires_grad=False)
elif args.xlnet:
    target_dir = 'output/Xlnet/'
    bertmodel = XlnetModel(requires_grad=False)
elif args.xlnet_pj:
    target_dir = 'output/Xlnet-pj/'
    bertmodel = XlnetModel(requires_grad=False)
else:
    raise Exception('Expect to choose a model.')

if args.dev:
    test_df = pd.read_csv(os.path.join('data/', "dev_pj.tsv"), sep='\t', header=None, names=['similarity', 's1'])
else:
    test_df = pd.read_csv(os.path.join('data/', "test_pj.tsv"), sep='\t', header=None, names=['similarity', 's1'])

max_seq_len = 50
batch_size = 32

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

Metric(test_df['similarity'], test_prediction['prediction'])

test_prediction.to_csv(os.path.join(target_dir, 'prediction.tsv'), sep='\t')
