import pymongo
import pickle as pkl
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SECONDS_PER_DAY = 24 * 3600
DATA_DIR = '../../score-data/CCMR/feateng/'

class GraphStore(object):
    def __init__(self):
        self.url = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(self.url)
     
class CCMRGraphStore(GraphStore):
    def __init__(self, rating_file, movie_info_dict):
        super(CCMRGraphStore, self).__init__()
        self.db = self.client['ccmr']
        self.user_coll = self.db.user
        self.item_coll = self.db.item
        
        self.user_num = 4920695
        self.item_num = 190129

        # input files
        self.rating_file = open(rating_file, 'r')
        with open(movie_info_dict, 'rb') as f:
            self.movie_info_dict = pkl.load(f)
        print('load movie info dict completed')

        # about time index
        self.time_idx_num = 21

    def gen_user_doc(self, uid):
        user_doc = {}
        user_doc['uid'] = uid
        for t in range(self.time_idx_num):
            user_doc['hist_%d'%t] = []
        return user_doc

    def gen_item_doc(self, iid):
        item_doc = {}
        item_doc['iid'] = iid
        item_doc['did'], item_doc['aid'], item_doc['gid'], item_doc['nid'] = self.movie_info_dict[str(iid)]
        for t in range(self.time_idx_num):
            item_doc['hist_%d'%t] = []
        return item_doc

    def construct_coll(self):
        collist = self.db.list_collection_names()
        if 'user' not in collist:
            print('begin initializing user collection')
            # construct user collection
            user_doc_list = []
            for i in range(1, self.user_num + 1):
                user_doc_list.append(self.gen_user_doc(i))
            print('initialize user doc list completed')

        if 'item' not in collist:
            print('begin initializing item collection')
            # construct item collection
            item_doc_list = []
            for i in range(self.user_num + 1, self.user_num + self.item_num + 1):
                item_doc_list.append(self.gen_item_doc(i))
            print('initialize item doc list completed')
        
        for line in self.rating_file:
            uid, iid, _, t_idx = line[:-1].split(',')
            user_doc_list[int(uid) - 1]['hist_%s'%(t_idx)].append(int(iid))
            item_doc_list[int(iid) - self.user_num - 1]['hist_%s'%(t_idx)].append(int(uid))
        print('user and item doc list completed')

        self.user_coll.insert_many(user_doc_list)
        print('user collection completed')
        self.item_coll.insert_many(item_doc_list)
        print('item collection completed')

    def cal_stat(self, user_hist_file, item_hist_file):
        # calculate user doc
        hist_len_user = []
        cursor = self.user_coll.find({})
        for user_doc in cursor:
            for t in range(self.time_idx_num):
                hist_len_user.append(len(user_doc['hist_%d'%(t)]))
        
        arr = np.array(hist_len_user)
        print('max user slice hist len: {}'.format(np.max(arr)))
        print('min user slice hist len: {}'.format(np.min(arr)))
        print('null slice per user: {}'.format(arr[arr == 0].size / self.user_num))
        print('small(<=5) slice per user: {}'.format(arr[arr <= 5].size / self.user_num))
        print('mean user slice(not null) hist len: {}'.format(np.mean(arr[arr > 0])))

        arr = arr.reshape(-1, self.time_idx_num)
        arr = np.sum(arr, axis=0)
        print(arr)

        
        print('-------------------------------------')
        # calculate item doc
        hist_len_item = []
        cursor = self.item_coll.find({})
        for item_doc in cursor:
            for t in range(self.time_idx_num):
                hist_len_item.append(len(item_doc['hist_%d'%(t)]))
        arr = np.array(hist_len_item)
        print('max item hist len: {}'.format(np.max(arr)))
        print('min item hist len: {}'.format(np.min(arr)))
        print('null per item: {}'.format(arr[arr == 0].size / self.item_num))
        print('small(<=5) per item: {}'.format(arr[arr <= 5].size / self.item_num))
        print('mean item hist(not null) len: {}'.format(np.mean(arr[arr > 0])))
        
        arr = arr.reshape(-1, self.time_idx_num)
        arr = np.sum(arr, axis=0)
        print(arr)


if __name__ == "__main__":
    # For CCMR
    gs = CCMRGraphStore(DATA_DIR + 'remap_rating_pos.csv', DATA_DIR + 'remap_movie_info_dict.pkl')
    # gs.construct_coll()
    gs.cal_stat(DATA_DIR + 'user_hist_len_distri.png', DATA_DIR + 'item_hist_len_distri.png')
