import random
import pymongo
import pickle as pkl
import time
import numpy as np

NEG_SAMPLE_NUM = 9
MAX_LEN = 80
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'

# CCMR dataset parameters
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129

class GraphLoader(object):
    def __init__(self, time_slice_num, db_name, user_neg_dict_file, obj_per_time_slice,
                 user_fnum, item_fnum, user_feat_dict_file = None, item_feat_dict_file = None):
        self.url = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(self.url)
        self.db = self.client[db_name]
        self.user_coll = self.db.user
        self.item_coll = self.db.item
        self.user_num = self.user_coll.find().count()
        self.item_num = self.item_coll.find().count()

        self.obj_per_time_slice = obj_per_time_slice
        with open(user_neg_dict_file, 'rb') as f:
            self.user_neg_dict = pkl.load(f)
        self.time_slice_num = time_slice_num

        self.user_fnum = user_fnum
        self.item_fnum = item_fnum
        self.user_feat_dict = None
        self.item_feat_dict = None
        # side information dict
        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        if item_feat_dict_file != None:
            with open(item_feat_dict_file, 'rb') as f:
                self.item_feat_dict = pkl.load(f)
        print('initial completed')

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

    def gen_target_file(self, pred_time, neg_sample_num, target_file):
        target_lines = []
        cursor = self.user_coll.find({})
        for user_doc in cursor:
            if user_doc['hist_%d'%(pred_time)] != []:
                uid = user_doc['uid']
                pos_iids = user_doc['hist_%d'%(pred_time)]
                for pos_iid in pos_iids:
                    target_lines.append(','.join([str(uid), str(pos_iid), str(1)]) + '\n')
                    neg_iids = self.gen_user_neg_items(uid, neg_sample_num, self.user_num + 1, self.user_num + self.item_num)
                    for neg_iid in neg_iids:
                        target_lines.append(','.join([str(uid), str(neg_iid), str(0)]) + '\n')
        with open(target_file, 'w') as f:
            f.writelines(target_lines)
        print('generate {} completed'.format(target_file))


    def gen_history(self, start_uid, start_iid, pred_time):
        user_1hop = []
        item_1hop = []
        user_2hop = []
        item_2hop = []

        start_user_doc = self.user_coll.find_one({'uid': start_uid})
        start_item_doc = self.item_coll.find_one({'iid': start_iid})
        for t in range(pred_time):
            user_1hop_list = start_user_doc['hist_%d'%(t)] #[iid1, iid2, ...]
            item_1hop_list = start_item_doc['hist_%d'%(t)] #[uid1, uid2, ...]
            # if too long
            if len(user_1hop_list) > MAX_LEN:
                user_1hop_list = np.random.choice(user_1hop_list, MAX_LEN, False).tolist()
            if len(item_1hop_list) > MAX_LEN:
                item_1hop_list = np.random.choice(item_1hop_list, MAX_LEN, False).tolist()

            # gen user 2 hops history
            if user_1hop_list == []:
                user_1hop.append(np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int))
                user_2hop.append(np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int))
            else:
                user_2hop_candi = []
                p_distri = []
                for iid in user_1hop_list:
                    item_doc = self.item_coll.find_one({'iid': iid})
                    degree = len(item_doc['hist_%d'%(t)])
                    for uid in item_doc['hist_%d'%(t)]:
                        if uid != start_uid:
                            user_2hop_candi.append(uid)
                            p_distri.append(float(1/(degree - 1)))
                p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                user_2hop_list = np.random.choice(user_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()

                if len(user_1hop_list) >= self.obj_per_time_slice:
                    user_1hop_list = np.random.choice(user_1hop_list, self.obj_per_time_slice, replace = False).tolist()
                else:
                    user_1hop_list = user_1hop_list + np.random.choice(user_1hop_list, self.obj_per_time_slice - len(user_1hop_list)).tolist()
                
                user_1hop_t = []
                for iid in user_1hop_list:
                    if self.item_feat_dict != None:
                        user_1hop_t.append([iid] + self.item_feat_dict[str(iid)])
                    else:
                        user_1hop_t.append([iid])
                user_1hop.append(user_1hop_t)
                
                user_2hop_t = []
                for uid in user_2hop_list:
                    if self.user_feat_dict != None:
                        user_2hop_t.append([uid] + self.user_feat_dict[str(uid)])
                    else:
                        user_2hop_t.append([uid])
                user_2hop.append(user_2hop_t)
            
            # gen item 2 hops history
            if item_1hop_list == []:
                item_1hop.append(np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int))
                item_2hop.append(np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int))
            else:
                item_2hop_candi = []
                p_distri = []
                for uid in item_1hop_list:
                    user_doc = self.user_coll.find_one({'uid': uid})
                    degree = len(user_doc['hist_%d'%(t)])
                    for iid in user_doc['hist_%d'%(t)]:
                        if iid != start_iid:
                            item_2hop_candi.append(iid)
                            p_distri.append(float(1/(degree - 1)))
                p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                item_2hop_list = np.random.choice(item_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                
                if len(item_1hop_list) >= self.obj_per_time_slice:
                    item_1hop_list = np.random.choice(item_1hop_list, self.obj_per_time_slice, replace = False).tolist()
                else:
                    item_1hop_list = item_1hop_list + np.random.choice(item_1hop_list, self.obj_per_time_slice - len(item_1hop_list)).tolist()

                item_1hop_t = []
                for uid in item_1hop_list:
                    if self.user_feat_dict != None:
                        item_1hop_t.append([uid] + self.user_feat_dict[str(uid)])
                    else:
                        item_1hop_t.append([uid])
                item_1hop.append(item_1hop_t)
                
                item_2hop_t = []
                for iid in item_2hop_list:
                    if self.item_feat_dict != None:
                        item_2hop_t.append([iid] + self.item_feat_dict[str(iid)])
                    else:
                        item_2hop_t.append([iid])
                item_2hop.append(item_2hop_t)
        return user_1hop, user_2hop, item_1hop, item_2hop

if __name__ == "__main__":
    graph_loader = GraphLoader(TIME_SLICE_NUM_CCMR, 'ccmr', DATA_DIR_CCMR + 'user_neg_dict.pkl', OBJ_PER_TIME_SLICE_CCMR, 1, 5)
    # graph_loader.gen_target_file(TIME_SLICE_NUM_CCMR - 2, NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_train.txt')
    # graph_loader.gen_target_file(TIME_SLICE_NUM_CCMR - 1, NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_test.txt')
    t = time.time()
    for uid in [4920679]:
        for iid in [5010458]:
            graph_loader.gen_history(uid, iid, 39)
    print('time: {}'.format((time.time() - t)))
