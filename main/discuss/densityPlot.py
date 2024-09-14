# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%

def rgba(r, g, b, a):
    return (r / 255, g / 255, b / 255, a / 1.0)

def render_plt(datas, series_labels, colors, x_label='', y_label='', limit=None):
    # 将数据放入 DataFrame
    df_list = {}
    for data, label in zip(datas, series_labels):
        df_list[label] = data
    
    df = pd.DataFrame(df_list)
    
    # 使用 melt 函数将 DataFrame 转换为长格式
    df_melt = df.melt(var_name='Series', value_name='Value')

    palette = {}
    for label, color in zip(series_labels, colors):
        palette[label] = color
    
    # 使用 FacetGrid 绘制每个系列的密度图
    g = sns.FacetGrid(df_melt, row='Series', aspect=4, height=2, palette=palette)
    g.map_dataframe(sns.kdeplot, 'Value', fill=True, common_norm=False)

    # 设置颜色
    for ax, (series, color) in zip(g.axes.flatten(), palette.items()):
        sns.kdeplot(data=df_melt[df_melt['Series'] == series], x='Value', fill=True, ax=ax, color=color)
    
    if limit is not None:
        g.set(xlim=limit)
    # 设置标题和标签
    g.set_titles('{row_name}')
    g.set_axis_labels(x_label, y_label)

    plt.show()

# %%
def compute_data(path_list, limit_range=0, print_gold=False):
    datas = []
    for path in path_list:
        with open(path) as f:
            ori_data = f.readlines()

        data = []
        for line in ori_data:
            pred, gold = line.strip().split('\t')
            pred = float(pred)
            if not print_gold:
                if float(gold) >= limit_range:
                    data.append(pred)
            else:
                if float(gold) >= limit_range:
                    data.append(float(gold))
        datas.append(np.array(data))
    
    return datas

datas = compute_data(['/home/lpc/repos/sTextSim/data_record/SimCSE_TransferDAWoScore_unsup/predict_gold1375.csv',
                      '/home/lpc/repos/sTextSim/data_record/GCSE_TransferDAWoScore6.4_unsup/predict_gold750.csv'], 4)
render_plt(datas, ['w/o decay', 'decay=6.4'], [rgba(0, 90, 158, 1), rgba(0, 90, 158, 1)], x_label='The distribution of cosine similarity for sentence pairs of STS-B dev set.', limit=(0, 1.2))

# %%
