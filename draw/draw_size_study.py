# coding:utf-8
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def draw_ab_study(fig_path='size_study.pdf'):
    x = [1, 5, 10, 15, 20]

    metric_dict = {
        'CCMR':   [[0.6743, 0.7001,	0.6912,	0.6922,	0.6897], [0.8932, 0.9118, 0.9101, 0.9011, 0.9008], [0.9212,	0.9583,	0.9555,	0.9498,	0.9423], [0.7876, 0.8161, 0.8131, 0.8111, 0.8109], [0.7999,	0.8281,	0.8239,	0.8232,	0.8221], [0.7688, 0.7913, 0.7911, 0.7895, 0.7829]],
        'Tmall':  [[0.4469, 0.4723,	0.4764,	0.4696,	0.4675], [0.6459, 0.6788, 0.6806, 0.6731, 0.6693], [0.7483,	0.7589,	0.7632,	0.7619,	0.7607], [0.5503, 0.5794, 0.5842, 0.5809, 0.5785], [0.5876,	0.6091,	0.6109,	0.6087,	0.6052], [0.5411, 0.5658, 0.5734, 0.5698, 0.5611]],
        'Taobao': [[0.2013, 0.2291,	0.2431,	0.2388,	0.2218], [0.4589, 0.4795, 0.4991, 0.4889, 0.4792], [0.6179,	0.6288,	0.6467,	0.6319,	0.6299], [0.3112, 0.3415, 0.3590, 0.3457, 0.3409], [0.3699,	0.4068,	0.4112,	0.4081,	0.4027], [0.3315, 0.3679, 0.3786, 0.3673, 0.3648]]
    }

    labels = ['HR@1', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'MRR']
    color = ['r-', 'b--', 'm-', 'y--', 'g--', 'c-.']
    datasets = ['CCMR', 'Tmall', 'Taobao']

    plt.figure(figsize=(18, 6))
    for i in range(len(datasets)):
        plt.subplot(1, 3, i + 1)
        ys = metric_dict[datasets[i]]
        for j in range(len(labels)):
            plt.plot(x, ys[j], color[j], label=labels[j])
        plt.legend(labels, loc='upper right', prop={'size': 12})
        plt.title('Performance Comparison on %s Dataset' % (datasets[i]))
        plt.xlabel(r'Size of Interaction Set $Num$')
        plt.ylabel('metric')

    plt.tight_layout()
    plt.savefig(fig_path)

if __name__ == "__main__":
    draw_ab_study()