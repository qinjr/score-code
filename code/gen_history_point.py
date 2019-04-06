import random
import pickle as pkl
import time
import numpy as np
import datetime

SECONDS_PER_DAY = 24 * 3600

# CCMR dataset parameters
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
START_TIME_CCMR = 1116432000
START_TIME_IDX_CCMR = 30
TIME_DELTA_CCMR = 90

def gen_user_item_hist_dict_ccmr(hist_file, user_hist_dict_file, item_hist_dict_file):
    user_hist_dict = {}
    item_hist_dict = {}

    # load and construct dicts
    with open(hist_file, 'r') as f:
        for line in f:
            uid, iid, _, time_str = line[:-1].split(',')
            time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
            time_idx = int((time_int - START_TIME_CCMR) / (SECONDS_PER_DAY * TIME_DELTA_CCMR))
            if time_idx < START_TIME_IDX_CCMR:
                continue
            if uid not in user_hist_dict:
                user_hist_dict[uid] = [(iid, time_int)]
            else:
                user_hist_dict[uid].append((iid, time_int))
            if iid not in item_hist_dict:
                item_hist_dict[iid] = [(uid, time_int)]
            else:
                item_hist_dict[iid].append((uid, time_int))
        print('dicts construct completed')

    # sort by time
    for uid in user_hist_dict.keys():
        user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
    for iid in item_hist_dict.keys():
        item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
    print('sort completed')

    # new dict
    user_hist_dict_sort = {}
    item_hist_dict_sort = {}
    for uid in user_hist_dict.keys():
        user_hist_dict[uid] = [tup[0] for tup in user_hist_dict[uid]]
    for iid in item_hist_dict.keys():
        item_hist_dict[iid] = [tup[0] for tup in item_hist_dict[iid]]
    print('new dict completed')

    # dump
    with open(user_hist_dict_file, 'wb') as f:
        pkl.dump(user_hist_dict_sort, f)
    with open(item_hist_dict_file, 'wb') as f:
        pkl.dump(item_hist_dict_sort, f)


if __name__ == "__main__":
    gen_user_item_hist_dict_ccmr(DATA_DIR_CCMR + 'rating_pos.csv', DATA_DIR_CCMR + 'user_hist_dict.pkl', DATA_DIR_CCMR + 'item_hist_dict.pkl')

