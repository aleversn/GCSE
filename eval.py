# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from main.evaluation import *
from main.models.gcse import GCSE

model_path = ''
tokenizer_path = ''
model = GCSE(from_pretrained=model_path,
                                pooler_type='cls')

main([
    '--model_name_or_path', model_path,
    '--tokenizer_path', tokenizer_path
], model)

# %%
