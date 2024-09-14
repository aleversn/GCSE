import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from main.trainers.gcse_trainer import Trainer
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained('/home/lpc/models/unsup-simcse-bert-base-uncased/')
trainer = Trainer(tokenizer=tokenizer,
                  from_pretrained='/home/lpc/models/unsup-simcse-bert-base-uncased/',
                  base_from_pretrained='/home/lpc/models/unsup-simcse-bert-base-uncased/',
                  data_present_path='./dataset/present.json',
                  max_seq_len=32,
                  hard_negative_weight=0,
                  batch_size=128,
                  temp=0.05,
                  data_name='SynCSEFullUn',
                  task_name='GCSE_SynCSEFullUn_unsup')

for i in trainer(num_epochs=3, lr=2e-5, gpu=[0], eval_call_step=lambda x: x % 125 == 0, save_per_call=True):
    a = i
