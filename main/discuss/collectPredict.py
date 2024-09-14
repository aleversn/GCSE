# %%
import numpy as np

obtain_indexes = [21, 51, 76, 80, 92]

with open('/home/lpc/repos/sTextSim/data_record/SimCSE_un/predict_gold0.csv') as f:
    lines = f.read().splitlines()

for id in obtain_indexes:
    line = lines[id - 1]
    line = line.split('\t')
    print(f'{float(line[0]) * 5}\t{line[1]}\t{np.sqrt((float(line[0]) * 5 - float(line[1])) ** 2)}\n')

# %%
