"""Simple training scripts for distilling T5"""

import time 
import torch
import re
import argparse
import os

import numpy as np
import torch.nn.functional as F

from torch import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

from .src.utils import tprint
from .src.data_utils import load_codex_generated

# OUTPUT_PATH = 'outputs/gsm8k/train_flan_t5_complex.txt'
DATA_PATH='outputs/gsm8k/'

def define_argument():
    ## add commandline arguments, initialized by the default configuration
    parser = argparse.ArgumentParser()   

    # general 
    parser.add_argument("--gpu_id", default='0', type=str)
    #   parser.add_argument("--output_path", default=OUTPUT_PATH, type=str)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--batch_size", default=10, type=int) 
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--data_mode", default='')
    parser.add_argument("--tune_mode", default="match_generation", type=str, 
        help="match_generation, match_distribution, contrastive")
    parser.add_argument("--generation_importance", default="uniform", type=str,
        help="uniform, emphasize_transition, emphasize_equation")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    return args

class GSM8KAugmentedDataset(Dataset):

    def __init__():
        return 

    def __len__():
        return

    def __getitem__():
        return


def train(args, model, train_dataloader, optimizer, scheduler):

    for e in args.num_epoch:

        # training epoch 
        for i, batch in enumerate(train_dataloader):
            # forward pass
            loss = model(batch)
            loss.backward()

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if i % args.log_interval == 0:
                tprint(f'Epoch: {e}, Iter: {i}, Loss: {loss.item()}')

        # validation on subset of training data 

        # validation on dev data
    return 

def eval(args, model, dev_dataset):
    return 

def main():
    ## arguments
    args = define_argument()

    ## data
    train_data_positive = load_codex_generated() # TODO: add path 
    # TODO: merge the generated data with the original data
    gsm8k = load_dataset('gsm8k', 'main') 
    dev_data_200 = ... # TODO: load the 200-sized dev data
    train_data_simple_200 = ... # 200 simple training data for validation, see if the model can remember the training subset 

    # train_data_negative = load_flan_t5_generated()
    dataset = GSM8KAugmentedDataset(train_data_positive)
    train_dataloader = DataLoader(dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True
                                  )

    ## model
    tprint('Loading the model ... ')
    start_time = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    # TODO: check if alpa can help speed up training / or DeepSpeed
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map='auto')
    tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) 
    scheduler = ... # TODO: add transformers scheduler 

    ## training 
    train(args, model, train_dataloader)

    ## Evaluation
    return 