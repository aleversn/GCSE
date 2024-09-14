# %% Combine Data
import os
from tqdm import tqdm

PREFIX = './data/nli_DA_extract/nli_train_sub_'
TOTAL = 6

all_data = []
for i in tqdm(range(TOTAL)):
    with open(PREFIX + ':' + str(i)) as f:
        data = f.readlines()
    all_data += data

with open(PREFIX, 'w') as f:
    for line in all_data:
        f.write(line)

# %% Construct KG
import json
from tqdm import tqdm

KG = {}
KG_map = {}

def extract_entity(info_json):
    for entity in info_json['entities']:
        if 'type' not in entity:
            entity['type'] = ''
        text, entity_type = entity['text'], entity['type']
        if entity_type == '':
            entity_type = 'none'
        if type(entity_type) is list:
            continue
        if entity_type not in KG_map:
            KG[entity_type] = []
            KG_map[entity_type] = {}
        if text in KG_map[entity_type]:
            continue
        KG_map[entity_type][text] = 1
        KG[entity_type].append(text)

with open('./data/nli_DA_extract/nli_train') as f:
    ori_data = f.readlines()

result = []
for line in tqdm(ori_data):
    line = line.strip()
    line = line.split('\t')
    info_json = json.loads(line[1])
    try:
        info_json = json.loads(info_json)
        for entity in info_json['entities']:
            if 'type' not in entity:
                entity['type'] = ''
            if type(entity['type']) is list:
                entity['type'] = ''
            if 'text' not in entity:
                info_json['entities'].remove(entity)
        for quantity in info_json['quantities']:
            if 'quantity' not in quantity:
                quantity['quantity'] = 1
            if int(quantity['quantity']) == 0:
                quantity['quantity'] = 1
    except:
        info_json = {
            'cls': 'story',
            'entities': [],
            'quantities': []
        }
    extract_entity(info_json)
    result.append([line[0], info_json])

with open('./data/nli_DA_extract/nli_train_extract.tsv', 'w') as f:
    for line in result:
        f.write(line[0] + '\t' + json.dumps(line[1], ensure_ascii=False) + '\n')

with open('./data/nli_DA_extract/kg.json', 'w') as f:
    json.dump(KG, f, ensure_ascii=False, indent=4)

# %%
# split extract data for generation
with open('./data/nli_DA_extract/nli_train_extract.tsv') as f:
    extract_data = f.readlines()

split_nums = 6
length = len(extract_data)

for i in range(split_nums):
    with open('./data/nli_DA_generated/nli_train_extract_sub_{}.tsv'.format(i), 'w') as f:
        for line in extract_data[i * length // split_nums : (i + 1) * length // split_nums]:
            f.write(line)


# %%
