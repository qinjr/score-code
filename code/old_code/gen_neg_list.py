import numpy as np
import cPickle as pkl
import random
NEG_CNT = 20

def gen_neg_list_file(item_pool_file = '../../data/taobao/item_pool.pkl', target_file = '../../data/taobao/score-data/target-test.txt',
                      neg_list_file = '../../data/taobao/score-data/neg_list.txt'):
    # get item pool
    with open(item_pool_file) as f:
        item_pool = pkl.load(f)
        print('load item_pool completed')
    
    neg_list_lines = []
    with open(target_file) as f:
        for line in f:
            neg_list_line_items = []
            target_item_iid = line.split(',')[1]
            while len(neg_list_line_items) < NEG_CNT:
                while True:
                    idx = random.randint(0, len(item_pool) - 1)
                    if target_item_iid != item_pool[idx][0]:
                        neg_list_line_items.append(','.join([item_pool[idx][0], item_pool[idx][1]]))
                        break
            neg_list_line = '\t'.join(neg_list_line_items) + '\n'
            neg_list_lines.append(neg_list_line)
    
    with open(neg_list_file, 'w') as f:
        f.writelines(neg_list_lines)

if __name__ == '__main__':
    gen_neg_list_file()