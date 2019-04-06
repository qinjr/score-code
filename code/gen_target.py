import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing

NEG_SAMPLE_NUM = 9
WORKER_N = 5
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'

# CCMR dataset parameters
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129
USER_PER_COLLECTION = 1000
ITEM_PER_COLLECTION = 100
START_TIME_CCMR = 1116432000
START_TIME_IDX_CCMR = 30
TIME_DELTA_CCMR = 90
class TargetGen(object):
    def __init__(self, user_neg_dict_file, db_name):
        with open(user_neg_dict_file, 'rb') as f:
            self.user_neg_dict = pkl.load(f)  
        
        url = "mongodb://localhost:27017/"
        client = pymongo.MongoClient(url)
        db = client[db_name]
        self.user_num = USER_NUM_CCMR
        self.item_num = ITEM_NUM_CCMR
        
        user_coll_num = self.user_num // USER_PER_COLLECTION
        if self.user_num % USER_PER_COLLECTION != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // ITEM_PER_COLLECTION
        if self.item_num % ITEM_PER_COLLECTION != 0:
            item_coll_num += 1

        self.user_colls = [db['user_%d'%(i)] for i in range(user_coll_num)]
        self.item_colls = [db['item_%d'%(i)] for i in range(item_coll_num)]
        
    def gen_user_neg_items(self, uid, neg_sample_num, iid_start, iid_end):
        if str(uid) in self.user_neg_dict:
            user_neg_list = self.user_neg_dict[str(uid)]
            user_neg_list = [int(iid) for iid in user_neg_list]
        else:
            user_neg_list = []
        
        if len(user_neg_list) >= neg_sample_num:
            return user_neg_list[:neg_sample_num]
        else:
            for i in range(neg_sample_num - len(user_neg_list)):
                user_neg_list.append(random.randint(iid_start, iid_end))
            return user_neg_list

    def gen_target_file(self, neg_sample_num, target_file, pred_time):
        target_lines = []
        for user_coll in self.user_colls:
            cursor = user_coll.find({})
            for user_doc in cursor:
                # if user_doc['hist_%d'%(pred_time)] != []:
                if user_doc['1hop'][pred_time] != []:
                    uid = user_doc['uid']
                    # pos_iids = user_doc['hist_%d'%(pred_time)]
                    pos_iids = user_doc['1hop'][pred_time]
                    # for pos_iid in pos_iids:
                    pos_iid = pos_iids[0]
                    neg_iids = self.gen_user_neg_items(uid, neg_sample_num, self.user_num + 1, self.user_num + self.item_num)
                    neg_iids = [str(neg_iid) for neg_iid in neg_iids]
                    target_lines.append(','.join([str(uid), str(pos_iid)] + neg_iids) + '\n')
        with open(target_file, 'w') as f:
            f.writelines(target_lines)
        print('generate {} completed'.format(target_file))
    
    def gen_user_item_hist_dict_ccmr(self, hist_file, user_hist_dict_file, item_hist_dict_file, pred_time=40):
        user_hist_dict = {}
        item_hist_dict = {}

        # load and construct dicts
        with open(hist_file, 'r') as f:
            for line in f:
                uid, iid, _, time_str = line[:-1].split(',')
                uid = str(int(uid) + 1)
                iid = str(int(iid) + 1 + self.user_num)
                time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
                time_idx = int((time_int - START_TIME_CCMR) / (SECONDS_PER_DAY * TIME_DELTA_CCMR))
                if time_idx < START_TIME_IDX_CCMR:
                    continue
                if time_idx >= pred_time:
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
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)
    
    def filter_target_file(self, target_file, target_file_hot, target_file_cold, user_hist_dict_file):
        with open(user_hist_dict_file, 'rb') as f:
            user_hist_dict = pkl.load(f)
        newlines_hot = []
        newlines_cold = []
        with open(target_file, 'r') as f:
            for line in f:
                uid = line[:-1].split(',')[0]
                if uid in user_hist_dict:
                    newlines_hot.append(line)
                else:
                    newlines_cold.append(line)
        with open(target_file_hot, 'w') as f:
            f.writelines(newlines_hot)
        with open(target_file_cold, 'w') as f:
            f.writelines(newlines_cold)
        print('filter target file completed')   

if __name__ == '__main__':
    tg = TargetGen(DATA_DIR_CCMR + 'user_neg_dict.pkl', 'ccmr_1hop')
    tg.filter_target_file(DATA_DIR_CCMR + 'target_40.txt', DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'target_40_cold.txt', DATA_DIR_CCMR + 'user_hist_dict.pkl')
    
