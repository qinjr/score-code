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

def gen_user_hist_seq_file(in_file, out_file, user_hist_dict_file, max_len, mode):
    with open(user_hist_dict_file, 'rb') as f:
        user_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in user_hist_dict:
                user_hist_list = user_hist_dict[uid]
                if mode == 'test':
                    if len(user_hist_list) > max_len:
                        user_hist_list = user_hist_list[-max_len:]
                    else:
                        user_hist_list = user_hist_list
                elif mode == 'train':
                    if len(user_hist_list) > max_len:
                        user_hist_list = user_hist_list[-max_len:-1]
                    else:
                        user_hist_list = user_hist_list[:-1]
                        if user_hist_list == []:
                            user_hist_list.append('0')
            else:
                print('WRONG')
                exit(1)
            newlines.append(','.join(user_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)

def gen_user_neg_items(pop_items, neg_sample_num = NEG_SAMPLE_NUM):
    user_neg_list = []
    pop_items_len = len(pop_items)
    for i in range(neg_sample_num):
        user_neg_list.append(pop_items[random.randint(0, pop_items_len-1)])
    return user_neg_list

def gen_point_models_target_train(in_file, out_file, user_hist_dict_file, pop_items_file):
    with open(user_hist_dict_file, 'rb') as f:
        user_hist_dict = pkl.load(f)
    with open(pop_items_file, 'rb') as f:
        pop_items = pkl.load(f)
    
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in user_hist_dict:
                user_hist_list = user_hist_dict[uid]
                target_iid = user_hist_list[-1]
                neg_iids = gen_user_neg_items(pop_items)
                neg_iids = [str(iid) for iid in neg_iids]
                newlines.append(','.join([target_iid] + neg_iids) + '\n')
            else:
                print('WRONG')
                exit(1)
    with open(out_file, 'w') as f:
        f.writelines(newlines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # CCMR
        gen_point_models_target_train(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'target_train4point_model.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', DATA_DIR_CCMR + 'pop_items.pkl')
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'train_user_hist_seq.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', MAX_LEN_CCMR, 'train')
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'test_user_hist_seq.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', MAX_LEN_CCMR, 'test')
    elif dataset == 'taobao':
        # Taobao
        gen_point_models_target_train(DATA_DIR_Taobao + 'target_17_hot.txt', DATA_DIR_Taobao + 'target_train4point_model.txt', DATA_DIR_Taobao + 'user_hist_dict_17.pkl', DATA_DIR_Taobao + 'pop_items.pkl')
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_17_hot.txt', DATA_DIR_Taobao + 'train_user_hist_seq.txt', DATA_DIR_Taobao + 'user_hist_dict_17.pkl', MAX_LEN_Taobao, 'train')
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_17_hot.txt', DATA_DIR_Taobao + 'test_user_hist_seq.txt', DATA_DIR_Taobao + 'user_hist_dict_17.pkl', MAX_LEN_Taobao, 'test')
    elif dataset == 'tmall':
        # Tmall
        gen_point_models_target_train(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'target_train4point_model.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', DATA_DIR_Tmall + 'pop_items.pkl')
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'train_user_hist_seq.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', MAX_LEN_Tmall, 'train')
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'test_user_hist_seq.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', MAX_LEN_Tmall, 'test')
    else:
        print('WRONG DATASET: {}'.format(dataset))