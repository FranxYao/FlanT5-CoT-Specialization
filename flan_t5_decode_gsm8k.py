"""
Decode GSM8K training data using the T5 model.

nohup python -u flan_t5_decode_gsm8k.py\
  --gpu_id 2\
  --output_path outputs/gsm8k/train_flan_t5_complex.txt\
  --debug 0\
  --num_sample 50\
  &> logs/flan_t5_decode_gsm8k.log &

tail -f logs/flan_t5_decode_gsm8k.log

TODO: add evaluation code
"""

import time 
import torch
import re
import numpy as np
import argparse

from datetime import datetime
import pytz

from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

OUTPUT_PATH = 'outputs/gsm8k/train_flan_t5_complex.txt'
PROMPT_PATH = 'lib_prompt/prompt_complex.txt'

def tprint(str):
  timezone = pytz.timezone("America/Vancouver")
  timenow = datetime.now(timezone)
  currenttime= timenow.strftime("%m/%d/%Y, %H:%M:%S")
  print(currenttime + ' ' + str)
  return  

def define_argument():
  ## add commandline arguments, initialized by the default configuration
  parser = argparse.ArgumentParser()   

  # general 
  parser.add_argument("--gpu_id", default='0', type=str)
  parser.add_argument("--output_path", default=OUTPUT_PATH, type=str)
  parser.add_argument("--debug", default=0, type=int)
  parser.add_argument("--num_sample", default=50, type=int)
  parser.add_argument("--log_interval", default=10, type=int)
  
  args = parser.parse_args()
  device = 'cuda:' + args.gpu_id
  return args, device

def process_prompt_complex(prompt, question):
  """
  Append the question to the prompt.
  Add "Let's think step by step." to the end of the prompt.
  """
  prompt_q = prompt + '\nQuestion: ' + question + "\nLet's think step by step"
  return prompt_q

def flan_decode(model, tokenizer, prompt, question, device='cuda:0', num_sample=1):
  """
  Decode a question using the T5 model.
  Return the decoded answer and the per-token log probabilities.
  """

  prompt_q = process_prompt_complex(prompt, question)
  # import ipdb; ipdb.set_trace()
  input_ids = tokenizer(prompt_q, return_tensors="pt").input_ids.to(device)
  out_dicts, out_texts = [], [] 
  with torch.no_grad():
    for _ in range(num_sample):
      out_dict = model.generate(input_ids,
                              do_sample=True, 
                              max_length=256, 
                              output_scores=True, 
                              return_dict_in_generate=True
                              )
      out_text = tokenizer.decode(out_dict['sequences'][0])
      out_dicts.append(out_dict)
      out_texts.append(out_text)
  return out_dicts, out_texts

def write_output(tokenizer, out_dicts, out_texts, question, qid, answer, fout):
  """Write output to file.
  Output consists of 
    question
    gold answer 
    num_sample * decoded answer
    num_sample * per step top 5 probabilities
  """
  fout.write(('Question %d: ' % qid) + question + '\n')
  fout.write('Answer: ' + answer + '\n')
  for i, (out_dict, out_text) in enumerate(zip(out_dicts, out_texts)):
    fout.write(('Model output %d: ' % i) + out_text + '\n')
    fout.write('Per-step decode:\n')
    for s in out_dict['scores']:
      s = torch.softmax(s, dim=-1)
      # import ipdb; ipdb.set_trace()
      probs, inds = torch.topk(s, 5)
      toks = tokenizer.convert_ids_to_tokens(inds[0])
      for t, p in zip(toks, probs[0]): 
        fout.write('%s, %.4f\t' % (t, p))
      fout.write('  ||  ')
    fout.write('\n')

  fout.write('\n\n\n\n')
  return 

def main():
  args, device = define_argument()

  # load the dataset
  gsm8k = load_dataset('gsm8k', 'main')

  # load the model
  tprint('Loading the model ... ')
  start_time = time.time()
  tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
  model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl").to(device)
  tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

  # load the prompt
  prompt_complex = open(PROMPT_PATH).read()
  
  # decode the dataset
  if(args.debug): end_id = 10
  else: end_id = len(gsm8k['train'])

  tprint('Start decoding ... ')
  with open(args.output_path, 'w') as fout:
    start_time = time.time()

    for i, (q, a) in enumerate(zip(gsm8k['train']['question'][:end_id], gsm8k['train']['answer'][:end_id])):
      # pass
      # import ipdb; ipdb.set_trace()

      out_dicts, out_texts = flan_decode(model, tokenizer, prompt_complex, q, device, args.num_sample)
      write_output(tokenizer, out_dicts, out_texts, q, i, a, fout)
    
      if(i % args.log_interval == 0): 
        tprint('Decoded %d / %d questions. time %.1fs' % (i, end_id, time.time() - start_time))
  return 

if __name__ == '__main__':
  main()