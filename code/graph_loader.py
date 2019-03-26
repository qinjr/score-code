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
                 user_fnum, item_fnum, target_file, batch_size, pred_time,
                 user_feat_dict_file = None, item_feat_dict_file = None):
        self.url = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(self.url)
        self.db = self.client[db_name]
        self.user_coll = self.db.user
        self.item_coll = self.db.item
        self.user_cursor = self.user_coll.find({})
        self.item_cursor = self.item_coll.find({})
        self.user_docs = []
        self.item_docs = []
        for user_docs in self.user_cursor:
            self.user_docs.append(user_docs)
        for item_docs in self.item_cursor:
            self.item_docs.append(item_docs)
        print('load graph data completed')

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
        
        self.target_f = open(target_file, 'r')

        self.batch_size = batch_size
        self.pred_time = pred_time

        print('graph loader initial completed')
    
    def __iter__(self):
        return self

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

    def gen_target_file(self, neg_sample_num, target_file):
        target_lines = []
        cursor = self.user_coll.find({})
        for user_doc in cursor:
            if user_doc['hist_%d'%(self.pred_time)] != []:
                uid = user_doc['uid']
                pos_iids = user_doc['hist_%d'%(self.pred_time)]
                for pos_iid in pos_iids:
                    neg_iids = self.gen_user_neg_items(uid, neg_sample_num, self.user_num + 1, self.user_num + self.item_num)
                    neg_iids = [str(neg_iid) for neg_iid in neg_iids]
                    target_lines.append(','.join([str(uid), str(pos_iid)] + neg_iids) + '\n')
        with open(target_file, 'w') as f:
            f.writelines(target_lines)
        print('generate {} completed'.format(target_file))

    def gen_user_history(self, start_uid):
        user_1hop = []
        user_2hop = []
        
        start_user_doc = self.user_coll.find_one({'uid': start_uid})
        for t in range(self.pred_time):
            user_1hop_list = start_user_doc['hist_%d'%(t)] #[iid1, iid2, ...]
            
            # if too long
            if len(user_1hop_list) > MAX_LEN:
                user_1hop_list = np.random.choice(user_1hop_list, MAX_LEN, False).tolist()
            
            # gen user 2 hops history
            if user_1hop_list == []:
                user_1hop.append(np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist())
                user_2hop.append(np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist())
            else:
                # deal with 1hop
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

                # deal with 2hop
                user_2hop_candi = []
                p_distri = []
                for iid in user_1hop_list:
                    item_doc = self.item_docs[iid - self.user_num - 1]
                    degree = len(item_doc['hist_%d'%(t)])
                    for uid in item_doc['hist_%d'%(t)]:
                        if uid != start_uid:
                            user_2hop_candi.append(uid)
                            p_distri.append(float(1/(degree - 1)))
                p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                user_2hop_list = np.random.choice(user_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                if user_2hop_list != []:
                    user_2hop_t = []
                    for uid in user_2hop_list:
                        if self.user_feat_dict != None:
                            user_2hop_t.append([uid] + self.user_feat_dict[str(uid)])
                        else:
                            user_2hop_t.append([uid])
                    user_2hop.append(user_2hop_t)
                else:
                    user_2hop.append(np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist())
            
        return user_1hop, user_2hop

    def gen_item_history(self, start_iid):
        item_1hop = []
        item_2hop = []

        start_item_doc = self.item_coll.find_one({'iid': start_iid})

        for t in range(self.pred_time):
            item_1hop_list = start_item_doc['hist_%d'%(t)] #[uid1, uid2, ...]
            # if too long
            if len(item_1hop_list) > MAX_LEN:
                item_1hop_list = np.random.choice(item_1hop_list, MAX_LEN, False).tolist()
            
            # gen item 2 hops history
            if item_1hop_list == []:
                item_1hop.append(np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist())
                item_2hop.append(np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist())
            else:
                # deal with 1hop
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

                # deal with 2hop
                item_2hop_candi = []
                p_distri = []
                for uid in item_1hop_list:
                    user_doc = self.user_docs[uid - 1]
                    degree = len(user_doc['hist_%d'%(t)])
                    for iid in user_doc['hist_%d'%(t)]:
                        if iid != start_iid:
                            item_2hop_candi.append(iid)
                            p_distri.append(float(1/(degree - 1)))
                p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                if item_2hop_candi != []
                    item_2hop_list = np.random.choice(item_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                    item_2hop_t = []
                    for iid in item_2hop_list:
                        if self.item_feat_dict != None:
                            item_2hop_t.append([iid] + self.item_feat_dict[str(iid)])
                        else:
                            item_2hop_t.append([iid])
                    item_2hop.append(item_2hop_t)
                else:
                    item_2hop.append(np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist())

        return item_1hop, item_2hop

    def __next__(self):
        if self.batch_size % (1 + NEG_SAMPLE_NUM) != 0:
            print('batch size should be time of {}'.format(1 + NEG_SAMPLE_NUM))
            exit(1)
        line_num = int(self.batch_size / 10)

        user_1hop_batch = []
        user_2hop_batch = []
        item_1hop_batch = []
        item_2hop_batch = []
        target_user_batch = []
        target_item_batch = []
        label_batch = []

        for b in range(line_num):
            line = self.target_f.readline()
            if line == '':
                raise StopIteration
            line_list = line[:-1].split(',')
            uid = int(line_list[0])
            user_1hop, user_2hop = self.gen_user_history(uid)
            for i in range(1 + NEG_SAMPLE_NUM):
                iid = int(line_list[1 + i])
                item_1hop, item_2hop = self.gen_item_history(iid)
                user_1hop_batch.append(user_1hop)
                user_2hop_batch.append(user_2hop)
                item_1hop_batch.append(item_1hop)
                item_2hop_batch.append(item_2hop)
                target_user_batch.append(uid)
                target_item_batch.append(iid)
                if i == 0:
                    label_batch.append(1)
                else:
                    label_batch.append(0)
        return [user_1hop_batch, user_2hop_batch, item_1hop_batch, item_2hop_batch, target_user_batch, target_item_batch, label_batch]

    def change_pred_time(self, new_pred_time):
        self.pred_time = new_pred_time
        return


if __name__ == "__main__":
    graph_loader = GraphLoader(TIME_SLICE_NUM_CCMR, 
                            'ccmr', 
                            DATA_DIR_CCMR + 'user_neg_dict.pkl', 
                            OBJ_PER_TIME_SLICE_CCMR, 
                            1, 
                            5,
                            DATA_DIR_CCMR + 'target_train.txt',
                            100,
                            40, 
                            None, 
                            DATA_DIR_CCMR + 'remap_movie_info_dict.pkl')
    # graph_loader.gen_target_file(TIME_SLICE_NUM_CCMR - 2, NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_train.txt')
    # graph_loader.gen_target_file(TIME_SLICE_NUM_CCMR - 1, NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_test.txt')
    # for batch_data in graph_loader:
    #     t = time.time()
    #     print(batch_data[-1])
    #     print('batch time')
    for i in range(100):
        t = time.time()
        print(graph_loader.__next__())
        print('batch time: {}'.format(time.time() - t))
