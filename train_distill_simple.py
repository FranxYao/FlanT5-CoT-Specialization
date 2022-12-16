"""Simple training scripts for distilling T5

This script is used for verifying that distillation from Codex can help FlanT5 improve performance on GSM8K

After the verification, we will transfer the code to huggingface trainer

nohup python -u train_distill_simple.py\
    --gpu_id 0,1,2,3,6,7\
    --log_interval 2\
    --num_epoch 10\
    --num_warmup_steps 10\
    --grad_accum_steps 5\
    --save_steps 1000,3500\
    --lr 5e-4\
    &> logs/beta_0.0.1.0.log &

tail -f logs/beta_0.0.1.0.log
"""

import time 
import torch
import re
import argparse
import os

import numpy as np
import torch.nn.functional as F

# from torch import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup

from src.utils import tprint, kl_divergence
from src.data_utils import GSM8KCodexAugmentedDataset

# OUTPUT_PATH = 'outputs/gsm8k/train_flan_t5_complex.txt'

def define_argument():
    ## add commandline arguments, initialized by the default configuration
    parser = argparse.ArgumentParser()   

    # general 
    parser.add_argument("--gpu_id", default='0', type=str)
    #   parser.add_argument("--output_path", default=OUTPUT_PATH, type=str)
    parser.add_argument("--model_version", default="beta_0.0.1.0", type=str)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--batch_size", default=10, type=int) 
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--num_epoch", default=10, type=int)
    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--grad_accum_steps", default=5, type=int) # TODO: adaptive gradient accumulation
    parser.add_argument("--save_steps", default='', type=str) 
    parser.add_argument("--save_path", default='checkpoints/', type=str) 
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--base_model", default='google/flan-t5-xxl', type=str,
        help="") # TODO: add OPT 66B
    parser.add_argument("--data_mode", default='')
    parser.add_argument("--tune_mode", default="match_generation", type=str, 
        help="match_generation, match_distribution, contrastive")
    parser.add_argument("--generation_importance", default="uniform", type=str,
        help="uniform, emphasize_transition, emphasize_equation")

    args = parser.parse_args()
    if(args.save_steps != ''):
        args.save_steps = [int(step) for step in args.save_steps.split(',')]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    return args

def compute_loss(logits, teacher_dist, mask):
    """Compute loss for the model

    logits: [batch_size, seq_len, vocab_size]
    teacher_dist: [batch_size, seq_len, vocab_size], teacher distribution from Codex
    """
    kld = kl_divergence(teacher_dist, F.softmax(logits, dim=-1))
    loss = (kld * mask).sum() / mask.sum()
    return loss

def save(model, save_path):
    torch.save(model.state_dict(), save_path)
    return 

def train(args, tokenizer, model, dataset, train_batches, optimizer, scheduler):

    global_step = 0
    smoothed_loss = 0.
    for e in range(args.num_epoch):

        # training epoch 
        for i, batch in enumerate(train_batches):
            # TODO: seperate transition loss and in-step loss to check which part is hard to learn
            batch = dataset.process_batch(tokenizer, batch)

            out_dict = model(input_ids=batch['src_input_ids'],
                             decoder_input_ids=batch['tgt_input_ids'],
                             return_dict=True
                             )

            loss = compute_loss(logits=out_dict['logits'], 
                                teacher_dist=batch['tgt_targets'],
                                mask=batch['tgt_mask']
                                )

            # gradient accumulation
            smoothed_loss += loss.item()
            loss /= args.grad_accum_steps
            loss.backward()
            if ((i + 1) % args.grad_accum_steps == 0) or (i + 1 == len(train_batches)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() 
                global_step += 1
                smoothed_loss = smoothed_loss / args.grad_accum_steps
                if global_step % args.log_interval == 0:
                    tprint(f'Epoch: %d, Iter: %d, Global step %d, Lr: %.4g, Loss: %.4f' % (e, i, global_step, scheduler.get_last_lr()[0], smoothed_loss))
                    smoothed_loss = 0
            
            if(e == 0 and i in args.save_steps):
                save_path = args.save_path + args.model_version + '_epoch_%d_iter_%d.pt' % (e, i)
                tprint('Saving model at %s' % save_path)
                save(model, save_path)

        # validation on subset of training data 
        save_path = args.save_path + args.model_version + '_epoch_%d_end.pt' % e
        tprint('Epoch %d finished, saving model at %s' % (e, save_path))
        save(model, save_path)

        # validation on dev data
    return 

def eval(args, model, dev_dataset):
    return 

def main():
    ## arguments
    args = define_argument()

    ## data
    dataset = GSM8KCodexAugmentedDataset()
    # TODO: merge the generated data with the original data
    # gsm8k = load_dataset('gsm8k', 'main') 
    # dev_data_200 = ... # TODO: load the 200-sized dev data
    # train_data_simple_200 = ... # 200 simple training data for validation, see if the model can remember the training subset 
    # dev_data_multiarith_200 = ... # TODO: load the 200-sized multiarith dev data, see if the model can generalize to multiarith

    # TODO: put contrastive cases into **the same batch**
    train_batches = dataset.get_train_batches(batch_size=20,
                                        target_answer_label=1,
                                        questions=dataset.questions, 
                                        answers=dataset.answers, 
                                        per_step_probs=dataset.per_step_probs, 
                                        per_step_mask=dataset.per_step_mask, 
                                        prediction_labels=dataset.prediction_labels
                                        )

    ## model
    tprint('Loading the model ... ')
    start_time = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

    # TODO: check if alpa can help speed up training / or DeepSpeed
    # TODO: change the base model to be OPT 66B
    # TODO: add dropout 
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map='auto') 
    tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.num_warmup_steps, 
                                                num_training_steps=10*len(train_batches))

    ## training 
    train(args, tokenizer, model, dataset, train_batches, optimizer, scheduler)

    ## Evaluation
    return 

if __name__ == '__main__':
    main()