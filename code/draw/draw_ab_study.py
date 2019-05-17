# coding:utf-8
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def draw_ab_study(fig_path='ab_study.pdf'):
    metric_dict = {
        'SCoRe':  [[0.7001, 0.9118, 0.9583, 0.8161, 0.8281, 0.7913], [0.4764, 0.6806, 0.7632, 0.5842, 0.6109, 0.5734], [0.2431, 0.4991, 0.6467, 0.3590, 0.4112, 0.3786]],
        'No_Att': [[0.6710, 0.9113, 0.9509, 0.8028, 0.8157, 0.7741], [0.4439, 0.6574, 0.7620, 0.5555, 0.5893, 0.5467], [0.2206, 0.4785, 0.6216, 0.3570, 0.4001, 0.3575]],
        '1hop':   [[0.6601, 0.8919, 0.9296, 0.7887, 0.8011, 0.7621], [0.4648, 0.6718, 0.7632, 0.5733, 0.6029, 0.5633], [0.2193, 0.4698, 0.6172, 0.3441, 0.3991, 0.3456]],
        'GAT':    [[0.6718, 0.8996, 0.9358, 0.7986, 0.8104, 0.7724], [0.4649, 0.6447, 0.7287, 0.5596, 0.5866, 0.5543], [0.2202, 0.4767, 0.6201, 0.3621, 0.4017, 0.3498]],
        }

    metric_names = ['HR@1', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'MRR']
    labels = ['SCoRe', 'No_Att', '1hop', 'GAT']
    color = ['red', 'm', 'orange', 'green']
    datasets = ['CCMR', 'Tmall', 'Taobao']

    plt.figure(figsize=(6, 6))
    for i in range(len(datasets)):
        rects = []
        ind = 4 * np.arange(len(metric_names))  # the x locations for the groups
        width = 0.3  # the width of the bars
        ax = plt.subplot(len(datasets), 1, i + 1)
        for j in range(len(labels)):
            rect = ax.bar(ind + j * width, metric_dict[labels[j]][i], width, color=color[j])
            rects.append(rect)
        ax.set_ylabel('metric')
        ax.set_xlabel('model')
        if i == 0:
            ax.set_ylim(0.65, 1)
        elif i == 1:
            ax.set_ylim(0.4, 0.8)
        elif i == 2:
            ax.set_ylim(0.2, 0.7)
        else:
            ax.set_ylim(0, 1)
        ax.set_title('Performance Comparison on %s Dataset' % (datasets[i]))
        ax.set_xticks(ind + width)
        ax.set_xticklabels(metric_names)
        ax.grid(True, axis='y', alpha=0.2)
        ax.legend(rects, labels, prop={'size': 5})

    plt.tight_layout()
    plt.savefig(fig_path)

if __name__ == "__main__":
    draw_ab_study()