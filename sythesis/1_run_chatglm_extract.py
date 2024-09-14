# %%
import os
import sys
import math
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=0, help='n_gpu')
parser.add_argument('--num_clip', default=1, help='how many chunks you want to split the file.')
parser.add_argument('--n_file', default=0, help='generate the n-th chunks, and other will be skip.')
parser.add_argument('--skip', default=-1, help='skip the first n lines, the skip index is count from the start index of n-th chunks')
parser.add_argument('--file_name', default='/home/lpc/repos/ChatGLM_PEFT/data/nli_DA/nli_da.csv', help='file name')
parser.add_argument('--save_dir', default='./save', help='save dir')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/chatglm3-6b/', help='model from pretrained')
parser.add_argument('--use_vllm', default=True, help='use vllm or default model')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)
READ_FILE_NAME = args.file_name
SAVE_FILE_NAME = os.path.join(args.save_dir, os.path.basename(args.file_name))
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.use_vllm:
    llm = LLM(model=args.model_from_pretrained, trust_remote_code=True)
else:
    llm = AutoModel.from_pretrained(model_from_pretrained=args.model_from_pretrained, trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_from_pretrained=args.model_from_pretrained, trust_remote_code=True)

extraction_prompt = '''指令: 预测下述文本的主题类别、 包含的实体和量化信息
注意: 类别为['news', 'story', 'medical']中的一项, 量化信息是指文本中包含的有数值或单位的信息, 如'2GB', '三只杯子', 'two dogs'等
输出格式: json格式数据, 数据格式为: 
{
    cls: [], // 类别
    entities: [{text: '', type: ''}], // 实体, 'text'必须是Input文本中的子序列
    quantities: [{text: '', type: '', quantity: 0}] // 量化信息, 'text'必须是Input文本中的子序列
}
Input: 
'''

# with open('./prompt.txt', encoding='utf-8') as f:
#     extraction_prompt = f.read()

with open('{}'.format(READ_FILE_NAME), encoding='utf-8') as f:
    lines = f.readlines()

total = len(lines)
batch_num = math.ceil(total / int(args.num_clip))
skip_num = batch_num * int(args.n_file)
for idx, line in tqdm(enumerate(lines), total=len(lines)):
    if idx < skip_num:
        continue
    if idx < int(args.skip) + skip_num:
        continue
    line = line.strip()
    ask_content = extraction_prompt + line + '\n'
    if args.use_vllm:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.8, max_tokens=1000)
        outputs = llm.generate(ask_content, sampling_params, use_tqdm=False)
        answer = outputs[0].outputs[0].text
    else:
        res, _ = llm.chat(tokenizer, ask_content, history=[])
        answer = res['data']

    with open('{}:{}'.format(SAVE_FILE_NAME, args.n_file), encoding='utf-8', mode='a+') as f:
        f.write(line + '\t' + json.dumps(answer, ensure_ascii=False) + '\n')
    
    if idx > skip_num + batch_num:
        break

# %%
