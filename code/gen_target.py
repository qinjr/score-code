import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing
import datetime

NEG_SAMPLE_NUM = 9
SECONDS_PER_DAY = 24*3600
# CCMR dataset parameters
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129
USER_PER_COLLECTION_CCMR = 1000
ITEM_PER_COLLECTION_CCMR = 100
START_TIME_CCMR = 1116432000
START_TIME_IDX_CCMR = 30
TIME_DELTA_CCMR = 90

# Taobao dataset parameters
DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'
TIME_SLICE_NUM_Taobao = 9
OBJ_PER_TIME_SLICE_Taobao = 10
USER_NUM_Taobao = 984105
ITEM_NUM_Taobao = 4067842
USER_PER_COLLECTION_Taobao = 500
ITEM_PER_COLLECTION_Taobao = 1000
START_TIME_Taobao = int(time.mktime(datetime.datetime.strptime('2017-11-25', "%Y-%m-%d").timetuple()))
START_TIME_IDX_Taobao = 0
TIME_DELTA_Taobao = 1

# Tmall dataset parameters
DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'
TIME_SLICE_NUM_Tmall = 7
OBJ_PER_TIME_SLICE_Tmall = 10
USER_NUM_Tmall = 424170
ITEM_NUM_Tmall = 1090390
USER_PER_COLLECTION_Tmall = 200
ITEM_PER_COLLECTION_Tmall = 250
START_TIME_Tmall = int(time.mktime(datetime.datetime.strptime('2015-5-1', "%Y-%m-%d").timetuple()))
START_TIME_IDX_Tmall = 0
TIME_DELTA_Tmall = 15


class TargetGen(object):
    def __init__(self, user_neg_dict_file, db_name, user_num, item_num, user_per_collection,
                item_per_collection, start_time, start_time_idx, time_delta):
        if user_neg_dict_file != None:
            with open(user_neg_dict_file, 'rb') as f:
                self.user_neg_dict = pkl.load(f)  
        else:
            self.user_neg_dict = {}

        url = "mongodb://localhost:27017/"
        client = pymongo.MongoClient(url)
        db = client[db_name]
        self.user_num = user_num
        self.item_num = item_num
        
        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection

        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1

        self.user_colls = [db['user_%d'%(i)] for i in range(user_coll_num)]
        self.item_colls = [db['item_%d'%(i)] for i in range(item_coll_num)]
        
        self.start_time = start_time
        self.start_time_idx = start_time_idx
        self.time_delta = time_delta

    def gen_user_neg_items(self, uid, neg_sample_num, start_iid, end_iid):
        if str(uid) in self.user_neg_dict:
            user_neg_list = self.user_neg_dict[str(uid)]
        else:
            user_neg_list = []
        
        if len(user_neg_list) >= neg_sample_num:
            return user_neg_list[:neg_sample_num]
        else:
            for i in range(neg_sample_num - len(user_neg_list)):
                user_neg_list.append(str(random.randint(start_iid, end_iid)))
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
                    target_lines.append(','.join([str(uid), str(pos_iid)] + neg_iids) + '\n')
        with open(target_file, 'w') as f:
            # random.shuffle(target_lines)
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
                time_idx = int((time_int - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if time_idx < self.start_time_idx:
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

    def gen_user_item_hist_dict_taobao(self, hist_file, user_hist_dict_file, item_hist_dict_file, remap_dict_file, pred_time=8):
        user_hist_dict = {}
        item_hist_dict = {}
        
        with open(remap_dict_file, 'rb') as f:
            uid_remap_dict = pkl.load(f)
            iid_remap_dict = pkl.load(f)

        # load and construct dicts
        with open(hist_file, 'r') as f:
            for line in f:
                uid, iid, _, timestamp_str = line[:-1].split(',')
                uid = uid_remap_dict[uid]
                iid = iid_remap_dict[iid]

                timestamp = int(timestamp_str)
                time_idx = int((timestamp - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if int(time_idx) < self.start_time_idx:
                    continue
                if int(time_idx) >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, timestamp)]
                else:
                    user_hist_dict[uid].append((iid, timestamp))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, timestamp)]
                else:
                    item_hist_dict[iid].append((uid, timestamp))
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

    def gen_user_item_hist_dict_tmall(self, hist_file, user_hist_dict_file, item_hist_dict_file, remap_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}
        
        with open(remap_dict_file, 'rb') as f:
            uid_remap_dict = pkl.load(f)
            iid_remap_dict = pkl.load(f)

        # load and construct dicts
        with open(hist_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                uid, iid, cid, sid, bid, date, btypeid, aid, gid = line[:-1].split(',')
                uid = uid_remap_dict[uid]
                iid = iid_remap_dict[iid]

                time_int = int(time.mktime(datetime.datetime.strptime('2015'+date, "%Y%m%d").timetuple()))
                time_idx = int((time_int - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if int(time_idx) < self.start_time_idx:
                    continue
                if int(time_idx) >= pred_time:
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
    # CCMR
    # tg = TargetGen(DATA_DIR_CCMR + 'user_neg_dict.pkl', 'ccmr_1hop', user_num = USER_NUM_CCMR,
    #             item_num = ITEM_NUM_CCMR, user_per_collection = USER_PER_COLLECTION_CCMR,
    #             item_per_collection = ITEM_PER_COLLECTION_CCMR, start_time = START_TIME_CCMR, 
    #             start_time_idx = START_TIME_IDX_CCMR, time_delta = TIME_DELTA_CCMR)
    
    # # tg.gen_user_item_hist_dict_ccmr(DATA_DIR_CCMR + 'rating_pos.csv', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', DATA_DIR_CCMR + 'item_hist_dict_40.pkl', 40)
    # tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_40.txt', 40)
    # tg.filter_target_file(DATA_DIR_CCMR + 'target_40.txt', DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'target_40_cold.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl')
    
    # # Taobao
    # tg = TargetGen(None, 'taobao_1hop', user_num = USER_NUM_Taobao,
    #             item_num = ITEM_NUM_Taobao, user_per_collection = USER_PER_COLLECTION_Taobao,
    #             item_per_collection = ITEM_PER_COLLECTION_Taobao, start_time = START_TIME_Taobao, 
    #             start_time_idx = START_TIME_IDX_Taobao, time_delta = TIME_DELTA_Taobao)

    # # tg.gen_user_item_hist_dict_taobao(DATA_DIR_Taobao + 'filtered_user_behavior.txt', DATA_DIR_Taobao + 'user_hist_dict_8.pkl', DATA_DIR_Taobao + 'item_hist_dict_8.pkl', DATA_DIR_Taobao + 'remap_dict.pkl', 8)
    # tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Taobao + 'target_8.txt', 8)
    # tg.filter_target_file(DATA_DIR_Taobao + 'target_8.txt', DATA_DIR_Taobao + 'target_8_hot.txt', DATA_DIR_Taobao + 'target_8_cold.txt', DATA_DIR_Taobao + 'user_hist_dict_8.pkl')

    Tmall
    tg = TargetGen(None, 'tmall_1hop', user_num = USER_NUM_Tmall,
                item_num = ITEM_NUM_Tmall, user_per_collection = USER_PER_COLLECTION_Tmall,
                item_per_collection = ITEM_PER_COLLECTION_Tmall, start_time = START_TIME_Tmall, 
                start_time_idx = START_TIME_IDX_Tmall, time_delta = TIME_DELTA_Tmall)
    
    tg.gen_user_item_hist_dict_tmall(DATA_DIR_Tmall + 'joined_user_behavior.csv', DATA_DIR_Tmall + 'user_hist_dict_12.pkl', DATA_DIR_Tmall + 'item_hist_dict_12.pkl', DATA_DIR_Tmall + 'remap_dict.pkl', 12)
    tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Tmall + 'target_12.txt', 12)
    tg.filter_target_file(DATA_DIR_Tmall + 'target_12.txt', DATA_DIR_Tmall + 'target_12_hot.txt', DATA_DIR_Tmall + 'target_12_cold.txt', DATA_DIR_Tmall + 'user_hist_dict_12.pkl')
