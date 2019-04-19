import random
import pickle as pkl
import time
import numpy as np
import datetime

# CCMR dataset parameters
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 100

# Taobao dataset parameters
DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 300

# Tmall dataset parameters
DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'
MAX_LEN_Tmall = 150

def gen_user_hist_seq_file(in_file, out_file, user_hist_dict_file, max_len):
    with open(user_hist_dict_file, 'rb') as f:
        user_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in user_hist_dict:
                user_hist_list = user_hist_dict[uid]
                if len(user_hist_list) > max_len:
                    user_hist_list = user_hist_list[-max_len:]
            else:
                user_hist_list = ['0']
            newlines.append(','.join(user_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)

def gen_item_hist_seq_file(in_file, out_file, item_hist_dict_file, max_len):
    with open(item_hist_dict_file, 'rb') as f:
        item_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            iids = line[:-1].split(',')[1:]
            for iid in iids:
                if iid in item_hist_dict:
                    item_hist_list = item_hist_dict[iid]
                    if len(item_hist_list) > max_len:
                        item_hist_list = item_hist_list[-max_len:]
                else:
                    item_hist_list = ['0']
                newlines.append(','.join(item_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)


if __name__ == "__main__":
    # CCMR
    # gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_train.txt', DATA_DIR_CCMR + 'train_user_hist_seq.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', MAX_LEN_CCMR)
    # gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_test.txt', DATA_DIR_CCMR + 'test_user_hist_seq.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', MAX_LEN_CCMR)

    # gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_train.txt', DATA_DIR_CCMR + 'train_item_hist_seq.txt', DATA_DIR_CCMR + 'item_hist_dict_40.pkl', MAX_LEN_CCMR)
    # gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot_test.txt', DATA_DIR_CCMR + 'test_item_hist_seq.txt', DATA_DIR_CCMR + 'item_hist_dict_40.pkl', MAX_LEN_CCMR)


    # # Taobao
    # gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_17_hot_train.txt', DATA_DIR_Taobao + 'train_user_hist_seq.txt', DATA_DIR_Taobao + 'user_hist_dict_17.pkl', MAX_LEN_Taobao)
    # gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_17_hot_test.txt', DATA_DIR_Taobao + 'test_user_hist_seq.txt', DATA_DIR_Taobao + 'user_hist_dict_17.pkl', MAX_LEN_Taobao)

    # gen_item_hist_seq_file(DATA_DIR_Taobao + 'target_17_hot_train.txt', DATA_DIR_Taobao + 'train_item_hist_seq.txt', DATA_DIR_Taobao + 'item_hist_dict_17.pkl', MAX_LEN_Taobao)
    # gen_item_hist_seq_file(DATA_DIR_Taobao + 'target_17_hot_test.txt', DATA_DIR_Taobao + 'test_item_hist_seq.txt', DATA_DIR_Taobao + 'item_hist_dict_17.pkl', MAX_LEN_Taobao)

    # Tmall
    gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_10_hot.txt', DATA_DIR_Tmall + 'train_user_hist_seq_10.txt', DATA_DIR_Tmall + 'user_hist_dict_10.pkl', MAX_LEN_Tmall)
    gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'test_user_hist_seq_11.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', MAX_LEN_Tmall)

    gen_item_hist_seq_file(DATA_DIR_Tmall + 'target_10_hot.txt', DATA_DIR_Tmall + 'train_item_hist_seq_10.txt', DATA_DIR_Tmall + 'item_hist_dict_10.pkl', MAX_LEN_Tmall)
    gen_item_hist_seq_file(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'test_item_hist_seq_11.txt', DATA_DIR_Tmall + 'item_hist_dict_11.pkl', MAX_LEN_Tmall)


