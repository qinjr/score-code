import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing

NEG_SAMPLE_NUM = 9
MAX_LEN = 80
WORKER_N = 16
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
        
        # multi-thread
        self.work_q = multiprocessing.Queue(maxsize=self.time_slice_num)
        self.worker_n = WORKER_N
        self.worker_done = multiprocessing.Value('d', self.time_slice_num)
        self.thread_list = []
        self.node_1hop = [None] * self.time_slice_num
        self.node_2hop = [None] * self.time_slice_num

        for i in range(self.worker_n):
            thread = multiprocessing.Process(target=self.gen_node_neighbor)
            thread.daemon = True
            self.thread_list.append(thread)
            thread.start()

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

    def gen_node_neighbor(self):
        node_1hop = []
        node_2hop = []
        
        while self.work_q.empty() == False and self.worker_done.value != self.time_slice_num:
            start_node_id, node_type, time_slice = self.work_q.get()
            if node_type == 'user':
                start_node_doc = self.user_coll.find_one({'uid': start_node_id})
                node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()
                node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
                
                node_1hop_nei_docs = self.item_docs
                node_1hop_nei_fnum = self.item_fnum
                node_1hop_nei_feat_dict = self.item_feat_dict
                node_2hop_nei_docs = self.user_docs
                node_2hop_nei_fnum = self.user_fnum
                node_2hop_nei_feat_dict = self.user_feat_dict

                index_reduce = self.user_num

            elif node_type == 'item':
                start_node_doc = self.item_coll.find_one({'iid': start_node_id})
                node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
                node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()

                node_1hop_nei_docs = self.user_docs
                node_1hop_nei_fnum = self.user_fnum
                node_1hop_nei_feat_dict = self.user_feat_dict
                node_2hop_nei_docs = self.item_docs
                node_2hop_nei_fnum = self.item_fnum
                node_2hop_nei_feat_dict = self.item_feat_dict

                index_reduce = 0
            
            node_1hop_list = start_node_doc['hist_%d'%(pred_time)] #[iid1, iid2, ...]
            
            # gen node 2 hops history
            if node_1hop_list == []:
                node_1hop.append(node_1hop_dummy)
                node_2hop.append(node_2hop_dummy)
            else:
                # deal with 1hop
                if len(node_1hop_list) >= self.obj_per_time_slice:
                    node_1hop_list = np.random.choice(node_1hop_list, self.obj_per_time_slice, replace = False).tolist()
                else:
                    node_1hop_list = node_1hop_list + np.random.choice(node_1hop_list, self.obj_per_time_slice - len(node_1hop_list)).tolist()

                node_1hop_t = []
                for node_id in node_1hop_list:
                    if node_1hop_nei_feat_dict != None:
                        node_1hop_t.append([node_id] + node_1hop_nei_feat_dict[str(node_id)])
                    else:
                        node_1hop_t.append([node_id])
                self.node_1hop[time_slice] = node_1hop_t

                # deal with 2hop
                node_2hop_candi = []
                p_distri = []
                for node_id in node_1hop_list:
                    node_1hop_nei_doc = node_1hop_nei_docs[node_id - index_reduce - 1]
                    degree = len(node_1hop_nei_doc['hist_%d'%(time_slice)])
                    for node_2hop_id in node_1hop_nei_doc['hist_%d'%(time_slice)]:
                        if node_2hop_id != start_node_id:
                            node_2hop_candi.append(node_2hop_id)
                            p_distri.append(float(1/(degree - 1)))
                p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                node_2hop_list = np.random.choice(node_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                if node_2hop_list != []:
                    node_2hop_t = []
                    for node_2hop_id in node_2hop_list:
                        if node_2hop_nei_feat_dict != None:
                            node_2hop_t.append([node_2hop_id] + node_2hop_nei_feat_dict[str(node_2hop_id)])
                        else:
                            node_2hop_t.append([node_2hop_id])
                    self.node_2hop[time_slice] = node_2hop_t
                else:
                    self.node_2hop[time_slice] = node_2hop_dummy
            with self.worker_done.get_lock():
                self.worker_done.value += 1

    def gen_user_history(self, start_uid):
        while self.work_q.empty() == False:
            continue
        
        for i in range(self.time_slice_num):
            self.work_q.put((start_uid, 'user', i))
        with self.worker_done.get_lock():
            self.worker_done.value = 0
        while self.worker_done == self.time_slice_num and self.work_q.empty():
            node_1hop, node_2hop = self.node_1hop, self.node_2hop
            self.node_1hop = [None] * self.time_slice_num
            self.node_2hop = [None] * self.time_slice_num
            return node_1hop, node_2hop


    def gen_item_history(self, start_iid):
        while self.work_q.empty() == False:
            continue
        
        for i in range(self.time_slice_num):
            self.work_q.put((start_iid, 'item', i))
        with self.worker_done.get_lock():
            self.worker_done.value = 0
        while self.worker_done == self.time_slice_num and self.work_q.empty():
            node_1hop, node_2hop = self.node_1hop, self.node_2hop
            self.node_1hop = [None] * self.time_slice_num
            self.node_2hop = [None] * self.time_slice_num
            return node_1hop, node_2hop
        
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
    t = time.time()
    for batch_data in graph_loader:
        print('batch time: {}'.format(time.time() - t))
        break
        # t = time.time()
