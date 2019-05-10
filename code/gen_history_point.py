import random
import pickle as pkl
import time
import numpy as np
import datetime
import sys

NEG_SAMPLE_NUM = 9

# CCMR dataset parameters
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 300

# Taobao dataset parameters
DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 300

# Tmall dataset parameters
DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'
MAX_LEN_Tmall = 300

def gen_user_hist_seq_file(in_file, out_file, user_hist_dict_file, max_len):
    with open(user_hist_dict_file, 'rb') as f:
        user_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in user_hist_dict:
                user_hist_list = user_hist_dict[uid]
                # if mode == 'test':
                if len(user_hist_list) > max_len:
                    user_hist_list = user_hist_list[-max_len:]
                else:
                    user_hist_list = user_hist_list
            else:
                print('WRONG')
                exit(1)
            newlines.append(','.join(user_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)

def gen_item_hist_seq_file(in_file, out_file, item_hist_dict_file, max_len):
    with open(item_hist_dict_file, 'rb') as f:
        item_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in item_hist_dict:
                item_hist_list = item_hist_dict[uid]
                # if mode == 'test':
                if len(item_hist_list) > max_len:
                    item_hist_list = item_hist_list[-max_len:]
                else:
                    item_hist_list = item_hist_list
            else:
                print('WRONG')
                exit(1)
            newlines.append(','.join(item_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # CCMR
        # gen_point_models_target_train(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'target_train4point_model.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', DATA_DIR_CCMR + 'pop_items.pkl')
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_39_hot.txt', DATA_DIR_CCMR + 'train_user_hist_seq_39.txt', DATA_DIR_CCMR + 'user_hist_dict_39.pkl', MAX_LEN_CCMR)
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'test_user_hist_seq_40.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', MAX_LEN_CCMR)
        gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_39_hot.txt', DATA_DIR_CCMR + 'train_item_hist_seq_39.txt', DATA_DIR_CCMR + 'item_hist_dict_39.pkl', MAX_LEN_CCMR)
        gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'test_item_hist_seq_40.txt', DATA_DIR_CCMR + 'item_hist_dict_40.pkl', MAX_LEN_CCMR)
    elif dataset == 'taobao':
        # Taobao
        # gen_point_models_target_train(DATA_DIR_Taobao + 'target_17_hot.txt', DATA_DIR_Taobao + 'target_train4point_model.txt', DATA_DIR_Taobao + 'user_hist_dict_17.pkl', DATA_DIR_Taobao + 'pop_items.pkl')
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_7_hot.txt', DATA_DIR_Taobao + 'train_user_hist_seq_7.txt', DATA_DIR_Taobao + 'user_hist_dict_7.pkl', MAX_LEN_Taobao)
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_8_hot.txt', DATA_DIR_Taobao + 'test_user_hist_seq_8.txt', DATA_DIR_Taobao + 'user_hist_dict_8.pkl', MAX_LEN_Taobao)
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_7_hot.txt', DATA_DIR_Taobao + 'train_item_hist_seq_7.txt', DATA_DIR_Taobao + 'item_hist_dict_7.pkl', MAX_LEN_Taobao)
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_8_hot.txt', DATA_DIR_Taobao + 'test_item_hist_seq_8.txt', DATA_DIR_Taobao + 'item_hist_dict_8.pkl', MAX_LEN_Taobao)
    elif dataset == 'tmall':
        # Tmall
        # gen_point_models_target_train(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'target_train4point_model.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', DATA_DIR_Tmall + 'pop_items.pkl')
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_10_hot.txt', DATA_DIR_Tmall + 'train_user_hist_seq_10.txt', DATA_DIR_Tmall + 'user_hist_dict_10.pkl', MAX_LEN_Tmall)
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'test_user_hist_seq_11.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', MAX_LEN_Tmall)
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_10_hot.txt', DATA_DIR_Tmall + 'train_item_hist_seq_10.txt', DATA_DIR_Tmall + 'item_hist_dict_10.pkl', MAX_LEN_Tmall)
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'test_item_hist_seq_11.txt', DATA_DIR_Tmall + 'item_hist_dict_11.pkl', MAX_LEN_Tmall)

    else:
        print('WRONG DATASET: {}'.format(dataset))