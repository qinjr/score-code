import pymongo
import pickle as pkl
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SECONDS_PER_DAY = 24 * 3600
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
USER_PER_COLLECTION = 1000
ITEM_PER_COLLECTION = 1000
START_TIME = 30

class GraphStore(object):
    def __init__(self):
        self.url = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(self.url)
     
class CCMRGraphStore(GraphStore):
    def __init__(self, rating_file, movie_info_dict):
        super(CCMRGraphStore, self).__init__()
        self.db_1hop = self.client['ccmr_1hop']
        self.db_2hop = self.client['ccmr_2hop']
        
        self.user_num = 4920695
        self.item_num = 190129

        # input files
        self.rating_file = open(rating_file, 'r')
        with open(movie_info_dict, 'rb') as f:
            self.movie_info_dict = pkl.load(f)
        print('load movie info dict completed')

        # about time index
        self.time_slice_num = 41

    def gen_user_doc(self, uid):
        user_doc = {}
        user_doc['uid'] = uid
        user_doc['1hop'] = [[] for i in range(self.time_slice_num)]
        # for t in range(self.time_slice_num):
        #     user_doc['hist_%d'%t] = []
        return user_doc

    def gen_item_doc(self, iid):
        item_doc = {}
        item_doc['iid'] = iid
        item_doc['1hop'] = [[] for i in range(self.time_slice_num)]
        # item_doc['did'], item_doc['aid'], item_doc['gid'], item_doc['nid'] = self.movie_info_dict[str(iid)]
        # for t in range(self.time_slice_num):
        #     item_doc['hist_%d'%t] = []
        return item_doc

    def construct_coll_1hop(self):
        list_of_user_doc_list = []
        list_of_item_doc_list = []

        user_coll_num = self.user_num // USER_PER_COLLECTION
        if self.user_num % USER_PER_COLLECTION != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // ITEM_PER_COLLECTION
        if self.item_num % ITEM_PER_COLLECTION != 0:
            item_coll_num += 1

        for i in range(user_coll_num):
            user_doc_list = []
            for uid in range(i * USER_PER_COLLECTION + 1, (i + 1) * USER_PER_COLLECTION + 1):
                user_doc_list.append(self.gen_user_doc(uid))
            list_of_user_doc_list.append(user_doc_list)

        for i in range(item_coll_num):
            item_doc_list = []
            for iid in range(i * ITEM_PER_COLLECTION + 1 + self.user_num, (i + 1) * ITEM_PER_COLLECTION + 1 + self.user_num):
                item_doc_list.append(self.gen_item_doc(iid))
            list_of_item_doc_list.append(item_doc_list)

        for line in self.rating_file:
            uid, iid, _, t_idx = line[:-1].split(',')
            list_of_user_doc_list[(int(uid) - 1) // USER_PER_COLLECTION][(int(uid) - 1) % USER_PER_COLLECTION]['1hop'][int(t_idx)].append(int(iid))
            list_of_item_doc_list[(int(iid) - self.user_num - 1) // ITEM_PER_COLLECTION][(int(iid) - self.user_num - 1) % ITEM_PER_COLLECTION]['1hop'][int(t_idx)].append(int(uid))
            # user_doc_list[int(uid) - 1]['hist_%s'%(t_idx)].append(int(iid))
            # item_doc_list[int(iid) - self.user_num - 1]['hist_%s'%(t_idx)].append(int(uid))
        print('user and item doc list completed')

        for i in range(len(list_of_user_doc_list)):
            self.db_1hop['user_%d'%(i)].insert_many(list_of_user_doc_list[i])
        print('user collection completed')
        for i in range(len(list_of_item_doc_list)):
            self.db_1hop['item_%d'%(i)].insert_many(list_of_item_doc_list[i])
        print('item collection completed')
    
    def construct_coll_2hop(self):
        user_coll_num = self.user_num // USER_PER_COLLECTION
        if self.user_num % USER_PER_COLLECTION != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // ITEM_PER_COLLECTION
        if self.item_num % ITEM_PER_COLLECTION != 0:
            item_coll_num += 1
        
        user_colls = [self.db_1hop['user_%d'%i] for i in range(user_coll_num)]
        item_colls = [self.db_1hop['item_%d'%i] for i in range(item_coll_num)]

        all_user_docs = []
        all_item_docs = []
        for user_coll in user_colls:
            cursor = user_coll.find({})
            for user_doc in cursor:
                all_user_docs.append(user_doc)
        for item_coll in item_colls:
            cursor = item_coll.find({})
            for item_doc in cursor:
                all_item_docs.append(item_doc)
        print('loading 1hop graph data completed')
        
        # gen user 2hop
        print('user 2 hop gen begin')
        for i in range(user_coll_num):
            user_docs_block = []
            for uid in range(1 + i * USER_PER_COLLECTION, 1 + (i + 1) * USER_PER_COLLECTION):
                st = time.time()
                old_user_doc = all_user_docs[uid - 1]
                new_user_doc = {
                    'uid': uid,
                    # '1hop': old_user_doc['1hop'],
                    '2hop': [],
                    'degrees': []
                }
                for t in range(START_TIME):
                    new_user_doc['2hop'].append([])
                    new_user_doc['degrees'].append([])
                
                for t in range(START_TIME, self.time_slice_num):
                    iids = old_user_doc['1hop'][t]
                    uids_2hop = []
                    degrees_2hop = []
                    for iid in iids:
                        item_doc = all_item_docs[iid - 1 - self.user_num]
                        if len(item_doc['1hop'][t]) > 1:
                            uids_2hop += item_doc['1hop'][t]
                            degrees_2hop += [len(item_doc['1hop'][t])] * len(item_doc['1hop'][t])
                    new_user_doc['2hop'].append(uids_2hop)
                    new_user_doc['degrees'].append(degrees_2hop)
                user_docs_block.append(new_user_doc)
                print('user 2hop gen time: {}'.format(time.time() - st))
            self.db_2hop['user_%d'%i].insert_many(user_docs_block)
        print('user 2 hop gen completed')

        # gen item 2hop
        print('item 2 hop gen begin')
        for i in range(item_coll_num):
            item_docs_block = []
            for iid in range(1 + self.user_num + i * ITEM_PER_COLLECTION, 1 + self.user_num + (i + 1) * ITEM_PER_COLLECTION):
                st = time.time()
                old_item_doc = all_item_docs[iid - 1 - self.user_num]
                new_item_doc = {
                    'iid': iid,
                    # '1hop': old_item_doc['1hop'],
                    '2hop': [],
                    'degrees': []
                }
                for t in range(START_TIME):
                    new_item_doc['2hop'].append([])
                    new_item_doc['degrees'].append([])
                for t in range(self.time_slice_num):
                    uids = old_item_doc['1hop'][t]
                    iids_2hop = []
                    degrees_2hop = []
                    for uid in uids:
                        user_doc = all_user_docs[uid - 1]
                        if len(user_doc['1hop'][t]) > 1:
                            iids_2hop += user_doc['1hop'][t]
                            degrees_2hop += [len(user_doc['1hop'][t])] * len(user_doc['1hop'][t])
                    new_item_doc['2hop'].append(iids_2hop)
                    new_item_doc['degrees'].append(degrees_2hop)
                item_docs_block.append(new_item_doc)
                print('item 2hop gen time: {}'.format(time.time() - såt))
            self.db_2hop['item_%d'%i].insert_many(item_docs_block)
        print('item 2 hop gen completed')


    def cal_stat(self):
        # calculate user doc
        hist_len_user = []
        cursor = self.user_coll.find({})
        for user_doc in cursor:
            for t in range(self.time_slice_num):
                hist_len_user.append(len(user_doc['hist_%d'%(t)]))
        
        arr = np.array(hist_len_user)
        print('max user slice hist len: {}'.format(np.max(arr)))
        print('min user slice hist len: {}'.format(np.min(arr)))
        print('null slice per user: {}'.format(arr[arr == 0].size / self.user_num))
        print('small(<=5) slice per user: {}'.format(arr[arr <= 5].size / self.user_num))
        print('mean user slice(not null) hist len: {}'.format(np.mean(arr[arr > 0])))

        arr = arr.reshape(-1, self.time_slice_num)
        arr = np.sum(arr, axis=0)
        print(arr)

        
        print('-------------------------------------')
        # calculate item doc
        hist_len_item = []
        cursor = self.item_coll.find({})
        for item_doc in cursor:
            for t in range(self.time_slice_num):
                hist_len_item.append(len(item_doc['hist_%d'%(t)]))
        arr = np.array(hist_len_item)
        print('max item hist len: {}'.format(np.max(arr)))
        print('min item hist len: {}'.format(np.min(arr)))
        print('null per item: {}'.format(arr[arr == 0].size / self.item_num))
        print('small(<=5) per item: {}'.format(arr[arr <= 5].size / self.item_num))
        print('mean item hist(not null) len: {}'.format(np.mean(arr[arr > 0])))
        
        arr = arr.reshape(-1, self.time_slice_num)
        arr = np.sum(arr, axis=0)
        print(arr)


if __name__ == "__main__":
    # For CCMR
    gs = CCMRGraphStore(DATA_DIR_CCMR + 'remap_rating_pos_idx.csv', DATA_DIR_CCMR + 'remap_movie_info_dict.pkl')
    # gs.construct_coll_1hop()
    gs.construct_coll_2hop()
    # gs.cal_stat()
