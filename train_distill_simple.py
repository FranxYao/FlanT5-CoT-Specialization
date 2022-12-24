"""Simple training scripts for distilling T5

This script is used for verifying that distillation from Codex can help FlanT5 improve performance on GSM8K

After the verification, we will transfer the code to huggingface trainer


model_version=0.0.2.2
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'2,3\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log
"""

import time 
import torch
import re
import argparse
import os
import hydra

import numpy as np
import torch.nn.functional as F

# from torch import Dataset, DataLoader
from tqdm import tqdm
from torch import nn
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup, Adafactor

from src.utils import tprint, kl_divergence
from src.data_utils import GSM8KCodexAugmentedDataset, GSM8KCodexAugmentedInContextDataset
from omegaconf import DictConfig, OmegaConf


def compute_loss_match_dist(logits, teacher_dist, mask):
    """Compute loss for the model

    logits: [batch_size, seq_len, vocab_size]
    teacher_dist: [batch_size, seq_len, vocab_size], teacher distribution from Codex
    """
    kld = kl_divergence(teacher_dist, F.softmax(logits, dim=-1))
    loss = (kld * mask).sum() / mask.sum()
    return loss

def compute_loss_nll(lm_logits, targets, mask, device):
    loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets.to(device).view(-1), reduction='none')
    loss = (loss * mask.to(device).view(-1)).sum() / mask.sum()
    return loss

def train(args, tokenizer, model, dataset, train_batches, optimizer, scheduler=None):
    """Training loop"""
    device = model.device
    global_step = 0
    smoothed_loss = 0.
    tprint('Start trainig, %d batches in total' % len(train_batches))
    for e in range(args.num_epoch):

        # training epoch 
        for i, batch in enumerate(train_batches):
            batch = dataset.process_batch(tokenizer, batch)

            # TODO: check what device will be when using multi-gpu
            out_dict = model(input_ids=batch['questions'].to(device),
                             attention_mask=batch['question_mask'].to(device),
                             decoder_input_ids=batch['answers'].to(device),
                             decoder_attention_mask=batch['answer_mask'].to(device),
                             return_dict=True
                             )

            lm_logits = out_dict['logits']
            
            # TODO: seperate transition loss and in-step loss to check which part is hard to learn
            # TODO: distribution match
            loss = compute_loss_nll(lm_logits, batch['targets'], batch['answer_mask'], device)

            # gradient accumulation
            smoothed_loss += loss.item()
            loss /= args.grad_accum_steps
            loss.backward()
            if ((i + 1) % args.grad_accum_steps == 0) or (i + 1 == len(train_batches)):
                optimizer.step()
                optimizer.zero_grad()
                if(scheduler is not None): scheduler.step() 
                global_step += 1
                smoothed_loss = smoothed_loss / args.grad_accum_steps
                if global_step % args.log_interval == 0:
                    tprint(f'Epoch: %d, Iter: %d, Global step %d, Lr: %.4g, Loss: %.4f' % (e, i, global_step, scheduler.get_last_lr()[0], smoothed_loss))
                    smoothed_loss = 0
                # import ipdb; ipdb.set_trace()
            
            if(e == 0 and i in args.save_steps):
                save_path = args.save_path + args.model_version + '_epoch_%d_iter_%d' % (e, i)
                tprint('Saving model at %s' % save_path)
                model.save_pretrained(save_path)
                # save(model, save_path)

        # validation on subset of training data 
        save_path = args.save_path + args.model_version + '_epoch_%d_end' % e
        tprint('Model %s Epoch %d finished, saving model at %s' % (args.model_version, e, save_path))
        # save(model, save_path)
        model.save_pretrained(save_path)

        # validation on dev data
    return 

def eval(args, model, dev_dataset):
    return 

@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    ## arguments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # args = define_argument()

    ## data
    dataset = GSM8KCodexAugmentedInContextDataset(args.batch_sizes, args.data_formats)
    train_batches = dataset.get_train_batches()
    # import ipdb; ipdb.set_trace()

    ## model
    tprint('Loading the model ... ')
    start_time = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

    # TODO: check if alpa can help speed up training / or DeepSpeed
    # TODO: change the base model to be OPT 66B
    # TODO: add dropout 

    model = T5ForConditionalGeneration.from_pretrained(args.base_model) 
    model.parallelize(args.device_map)

    tokenizer.decoder_start_token_id = model.config.decoder_start_token_id # special treatment for T5

    tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.num_warmup_steps, 
                                                num_training_steps=10*len(train_batches))

    ## training 
    train(args, tokenizer, model, dataset, train_batches, optimizer, scheduler)

    return 

if __name__ == '__main__':
    main()