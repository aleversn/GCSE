# %% 引入arg parser
import os
import sys
import json
import random
import copy
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from tqdm import tqdm
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=0, help='n_gpu')
parser.add_argument('--n_file', default=0, help='generate the n-th chunks, and other will be skip.')
parser.add_argument('--skip', default=-1, help='skip the first n lines, the skip index is count from the start index of n-th chunks')
parser.add_argument('--file_name', default='/home/lpc/repos/ChatGLM_PEFT/data/nli_DA/nli_da.csv', help='file name')
parser.add_argument('--save_dir', default='./save', help='save dir')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/chatglm3-6b/', help='model from pretrained')
parser.add_argument('--use_vllm', default=False, help='use vllm or default model')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)
READ_FILE_NAME = args.file_name + ':' + args.n_file
SAVE_FILE_NAME = os.path.join(args.save_dir, os.path.basename(args.file_name) + ':' + args.n_file)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.use_vllm:
    llm = LLM(model=args.model_from_pretrained, trust_remote_code=True)
else:
    llm = AutoModel.from_pretrained(model_from_pretrained=args.model_from_pretrained, trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_from_pretrained=args.model_from_pretrained, trust_remote_code=True)

rewrite_prompt = '''Instruction: You are a great story teller, rewrites the input sentence in another way
Input: '''
fake_prompt = '''Instruction: You are a liar, you should rewrite the input sentence so that the NLI model would treat it as a opposite sample.
Rules: 1. if the sentence say no, you should say yes, 2. if someone loves something, you should say it doesn't lovee it. 3. if some is doing something, say it is doing another thing. 4. if the sentence is positive/negative, say it as negative/positive
Input: '''
replace_entity_prompt = '''Instruction: You are a great story teller, rewrites the input sentence, and change the entity '$text$' to another $type$ '$replace_text$'.
Input: '''
replace_quantity_prompt = '''Instruction: You are a great story teller, rewrites the input sentence, and change the quantity $quantity$ of '$text$' to $replace_quantity$.
Input: '''
reconstruct_prompt = '''Instruction: You are a great rewriter, and I want you to generate new sentence according to the classification, entities and quantities info provided by the json
Rules: you should aware that the new text in "quantities" should be rewrite follows the "quantity" value. e.g. "text": "A man", "quantity": 5 should rewrite as "five men"
metadata:
'''

def denoise(sample, keep_last_seg=False):
    sample = sample.replace('(Positive)', '')
    sample = sample.replace('(Negative)', '')
    sample = sample.replace('...  ', '')
    sample = sample.replace('...', '')
    if keep_last_seg:
        sample = sample.split('.')
        sample = sample[-2:]
        sample = '.'.join(sample)
    sample = sample.replace('Result (Plain Text):', '')
    sample = sample.replace('Result (Plain Text)', '')
    sample = sample.replace('(No)', '')
    sample = sample.replace('->', '')
    return sample

with open(READ_FILE_NAME, encoding='utf-8') as f:
    lines = f.readlines()

with open(os.path.join(args.save_dir, 'kg.json')) as f:
    kg = json.load(f)

def tamper_info(info_json, kg, enable_tamper=True):
    fake_info_json = copy.deepcopy(info_json)
    random.shuffle(fake_info_json['entities'])
    random.shuffle(fake_info_json['quantities'])
    if enable_tamper and len(fake_info_json['entities']) != 0:
        for entity in fake_info_json['entities']:
            if 'text' in entity and 'type' in entity and entity['type'] not in ['', 'none', 'o', 'O', 'None']:
                entity['text'] = random.choice(kg[entity['type']])
                break
    if enable_tamper and len(fake_info_json['quantities']) != 0:
        for quantity in fake_info_json['quantities']:
            quantity['quantity'] = random.randint(0, 1000)
            break
    return fake_info_json

for idx, line in tqdm(enumerate(lines), total=len(lines)):
    if idx < int(args.skip):
        continue
    line = line.strip().split('\t')
    info_json = json.loads(line[1])

    ask_content_1 = rewrite_prompt + '\n' + line[0] + '\nGenerate Result (Plain Text):\n'
    ask_content_2 = fake_prompt + '\n' + line[0] + '\nGenerate Result (Plain Text):\n'

    random.shuffle(info_json['entities'])
    random.shuffle(info_json['quantities'])
    selection = {
        'prompt_type': 'none'
    }
    if len(info_json['entities']) != 0:
        for entity in info_json['entities']:
            if 'text' in entity and 'type' in entity and entity['type'] not in ['', 'none', 'o', 'O', 'None']:
                selection['text'] = entity['text']
                selection['type'] = entity['type']
                selection['replace_text'] = random.choice(kg[entity['type']])
                selection['prompt_type'] = 'entity'
    
    if len(info_json['quantities']) != 0 and random.randint(0, 100) >= 50:
        for quantity in info_json['quantities']:
            selection['text'] = quantity['text']
            selection['quantity'] = quantity['quantity']
            selection['replace_quantity'] = random.randint(0, 1000)
            selection['prompt_type'] = 'quantity'
    
    if selection['prompt_type'] == 'none':
        selection['text'] = line[0][:10]
        selection['type'] = 'segments'
        selection['replace_text'] = random.choice(kg['none'])
        selection['prompt_type'] = 'entity'
    
    if selection['prompt_type'] == 'entity':
        this_replace_entity_prompt = replace_entity_prompt
        this_replace_entity_prompt = this_replace_entity_prompt.replace('$text$', selection['text'])
        this_replace_entity_prompt = this_replace_entity_prompt.replace('$type$', selection['type'])
        this_replace_entity_prompt = this_replace_entity_prompt.replace('$replace_text$', selection['replace_text'])
        ask_content_3 = this_replace_entity_prompt + line[0] + '\nGenerate Result (Plain Text):\n'
    else:
        this_replace_quantity_prompt = replace_quantity_prompt
        this_replace_quantity_prompt = this_replace_quantity_prompt.replace('$text$', selection['text'])
        this_replace_quantity_prompt = this_replace_quantity_prompt.replace('$quantity$', str(selection['quantity']))
        this_replace_quantity_prompt = this_replace_quantity_prompt.replace('$replace_quantity$', str(selection['replace_quantity']))
        ask_content_3 = this_replace_quantity_prompt + line[0] + '\nGenerate Result (Plain Text):\n'
    
    if random.randint(0, 100) >= 50:
        fake_json = tamper_info(info_json, kg)
        fake_json = json.dumps(fake_json, ensure_ascii=False, indent=4)
        ask_content_3 = reconstruct_prompt + fake_json + '\nGenerate Result (Plain Text):\n'
    
    inputs = [ask_content_1, ask_content_2, ask_content_3]
    
    if args.use_vllm:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.8, max_tokens=2 * len(ask_content_1))
        outputs = llm.generate([inputs], sampling_params, use_tqdm=False)
        
        for output in outputs:
            answer = output.outputs[0].text
            with open('{}'.format(SAVE_FILE_NAME), encoding='utf-8', mode='a+') as f:
                f.write(str(idx) + '\t' + line[0] + '\t' + answer.replace('\n', ' ') + '\n')
    else:
        for input_text in inputs:
            res = llm.chat(tokenizer, input_text, history=[], max_length=len(input_text) + 2 * len(line[0]))
            answer = res['data']
            try:
                answer = denoise(answer)
            except:
                pass
            with open('{}'.format(SAVE_FILE_NAME), encoding='utf-8', mode='a+') as f:
                f.write(str(idx) + '\t' + line[0] + '\t' + answer.replace('\n', ' ') + '\n')
    
    

# %%
