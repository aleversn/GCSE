# %%
import re
import random
from tqdm import tqdm
import json
import copy

import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from main.predictors.cse_predictor import Predictor
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/lpc/models/unsup-simcse-bert-base-uncased/')
pred = Predictor(tokenizer=tokenizer,
                  from_pretrained='/home/lpc/models/unsup-simcse-bert-base-uncased/',
                  max_seq_len=32,
                  hard_negative_weight=0,
                  batch_size=64,
                  temp=0.05)

PREFIX_DIR = '/home/lpc/repos/sTextSim/dataset/nli_DA/mix'
with open(f'{PREFIX_DIR}/full_generated.tsv') as f:
    lines = f.readlines()

scores = []
inputs = []

lines = [line.strip().split('\t') for line in lines]
for idx, line in tqdm(enumerate(lines), total=len(lines)):
    if len(line) < 3:
        line.append('word')
    id, ori, generation = line[0], line[1], line[2]
    inputs.append([ori, generation])

for output in pred.pred(inputs):
    scores += torch.diag(output.logits).tolist()

result = []
groups = {}
current_id = ''
group_id = -1
for idx, line in tqdm(enumerate(lines), total=len(lines)):
    id, ori, generation = line[0], line[1], line[2]
    generation = generation[:len(ori) * 3]
    if current_id != id:
        current_id = id
        group_id += 1
    
    pos_score = scores[idx]

    if group_id not in groups:
        groups[group_id] = {
            'ori': ori,
            'samples': []
        }
    groups[group_id]['samples'].append({
        'id': idx,
        'score': pos_score,
        'text': generation
    })

neg_from_other_batch = 0
iter = tqdm(groups.values())
for idx, group in enumerate(iter):
    text1 = group['ori']
    text2 = text1
    neg = groups[idx + 100]['ori'] if idx + \
            100 < len(groups.values()) else groups[idx - 100]['ori']
    for sample in group['samples']:
        if sample['score'] >= 18:
            text2 = sample['text']
            break
    
    exists_neg = False
    for sample in group['samples']:
        if sample['score'] < 15:
            neg = sample['text']
            exists_neg = True
            break
    
    iter.set_description(f'neg_from_other_batch: {neg_from_other_batch}')
    if not exists_neg:
        neg_from_other_batch += 1
        continue
    
    
    result.append({
        'id': sample['id'],
        'text1': text1,
        'text2': text2,
        'neg': neg
    })
    

with open(f'{PREFIX_DIR}/full_generated.jsonl', 'w') as f:
    for line in result:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')

# %%
