import random
import pickle as pkl
import time
import numpy as np
import datetime

SECONDS_PER_DAY = 24 * 3600

# CCMR dataset parameters
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 100

def gen_user_hist_seq_file(in_file, out_file, user_hist_dict_file):
    with open(user_hist_dict_file, 'rb') as f:
        user_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in user_hist_dict:
                user_hist_list = user_hist_dict[uid]
                if len(user_hist_list) > MAX_LEN_CCMR:
                    user_hist_list = user_hist_list[-MAX_LEN_CCMR:]
            else:
                user_hist_list = ['0']
            newlines.append(','.join(user_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)

def gen_item_hist_seq_file(in_file, out_file, item_hist_dict_file):
    with open(item_hist_dict_file, 'rb') as f:
        item_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            iids = line[:-1].split(',')[1:]
            for iid in iids:
                if iid in item_hist_dict:
                    item_hist_list = item_hist_dict[iid]
                    if len(item_hist_list) > MAX_LEN_CCMR:
                        item_hist_list = item_hist_list[-MAX_LEN_CCMR:]
                else:
                    item_hist_list = ['0']
                newlines.append(','.join(item_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)


if __name__ == "__main__":
    gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_train.txt', DATA_DIR_CCMR + 'train_user_hist_seq.txt', DATA_DIR_CCMR + 'user_hist_dict.pkl')
    gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_test.txt', DATA_DIR_CCMR + 'test_user_hist_seq.txt', DATA_DIR_CCMR + 'user_hist_dict.pkl')

    gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_train.txt', DATA_DIR_CCMR + 'train_item_hist_seq.txt', DATA_DIR_CCMR + 'item_hist_dict.pkl')
    gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_test.txt', DATA_DIR_CCMR + 'test_item_hist_seq.txt', DATA_DIR_CCMR + 'item_hist_dict.pkl')

