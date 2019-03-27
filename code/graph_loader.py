import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing

NEG_SAMPLE_NUM = 9
MAX_LEN = 80
WORKER_N = 8
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'

# CCMR dataset parameters
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129


class GraphLoader(object):
    # mongo client and load data
    url = "mongodb://localhost:27017/"
    client = pymongo.MongoClient(url)
    db = client['ccmr']
    user_coll = db.user
    item_coll = db.item
    user_cursor = user_coll.find({})
    item_cursor = item_coll.find({})
    user_docs = []
    item_docs = []
    for user_doc in user_cursor:
        user_docs.append(user_doc)
    for item_doc in item_cursor:
        item_docs.append(item_doc)
    
    with open(DATA_DIR_CCMR + 'user_neg_dict.pkl', 'rb') as f:
        user_neg_dict = pkl.load(f)
    print('load static data completed')

    def __init__(self, time_slice_num, db_name, user_neg_dict_file, obj_per_time_slice,
                 user_fnum, item_fnum, target_file, batch_size, pred_time,
                 user_feat_dict_file = None, item_feat_dict_file = None):
        self.user_num = GraphLoader.user_coll.find().count()
        self.item_num = GraphLoader.item_coll.find().count()
        
        self.obj_per_time_slice = obj_per_time_slice
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
        self.worker_begin = multiprocessing.Value('d', 0)
        self.complete = multiprocessing.Value('d', 0)
        self.work_cnt = multiprocessing.Value('d', 0)

        self.thread_list = []
        self.node_1hop = [None] * self.time_slice_num
        self.node_2hop = [None] * self.time_slice_num

        for i in range(self.worker_n):
            thread = multiprocessing.Process(target=self.gen_node_neighbor)#, args=[i])
            thread.daemon = True
            self.thread_list.append(thread)
            thread.start()

        print('graph loader initial completed')
    
    def __iter__(self):
        return self

    def gen_user_neg_items(self, uid, neg_sample_num, iid_start, iid_end):
        if str(uid) in GraphLoader.user_neg_dict:
            user_neg_list = GraphLoader.user_neg_dict[str(uid)]
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
        cursor = GraphLoader.user_coll.find({})
        for user_doc in cursor:
            if user_doc['hist_%d'%(pred_time)] != []:
                uid = user_doc['uid']
                pos_iids = user_doc['hist_%d'%(pred_time)]
                for pos_iid in pos_iids:
                    neg_iids = self.gen_user_neg_items(uid, neg_sample_num, self.user_num + 1, self.user_num + self.item_num)
                    neg_iids = [str(neg_iid) for neg_iid in neg_iids]
                    target_lines.append(','.join([str(uid), str(pos_iid)] + neg_iids) + '\n')
        with open(target_file, 'w') as f:
            f.writelines(target_lines)
        print('generate {} completed'.format(target_file))


    def gen_node_neighbor(self):
        while True:
            if not self.work_q.empty() and self.worker_begin.value == 1:
                start_node_id, node_type, time_slice = self.work_q.get()
                if node_type == 'user':
                    start_node_doc = GraphLoader.user_docs[start_node_id - 1] 
                    node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()
                    node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
                    
                    node_1hop_nei_type = 'item'
                    node_1hop_nei_fnum = self.item_fnum
                    node_1hop_nei_feat_dict = self.item_feat_dict
                    node_2hop_nei_feat_dict = self.user_feat_dict

                elif node_type == 'item':
                    start_node_doc = GraphLoader.item_docs[start_node_id - 1 - self.user_num] 
                    node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
                    node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()

                    node_1hop_nei_type = 'user'
                    node_1hop_nei_fnum = self.user_fnum
                    node_1hop_nei_feat_dict = self.user_feat_dict
                    node_2hop_nei_feat_dict = self.item_feat_dict
                
                node_1hop_list = start_node_doc['hist_%d'%(time_slice)] #[iid1, iid2, ...]

                # gen node 2 hops history
                if node_1hop_list == []:
                    self.node_1hop[time_slice] = node_1hop_dummy
                    self.node_2hop[time_slice] = node_2hop_dummy
                    with self.work_cnt.get_lock():
                        self.work_cnt.value += 1
                    # return node_1hop_dummy, node_2hop_dummy
                else:
                    # deal with 1hop
                    if len(node_1hop_list) >= self.obj_per_time_slice:
                        node_1hop_list = np.random.choice(node_1hop_list, self.obj_per_time_slice, replace = False).tolist()
                        node_1hop_list_unique = node_1hop_list
                    else:
                        node_1hop_list_unique = node_1hop_list
                        node_1hop_list = node_1hop_list + np.random.choice(node_1hop_list, self.obj_per_time_slice - len(node_1hop_list)).tolist()

                    node_1hop_t = []
                    for node_id in node_1hop_list:
                        if node_1hop_nei_feat_dict != None:
                            node_1hop_t.append([node_id] + node_1hop_nei_feat_dict[str(node_id)])
                        else:
                            node_1hop_t.append([node_id])

                    # deal with 2hop            
                    node_2hop_candi = []
                    p_distri = []
                    for node_id in node_1hop_list_unique:
                        if node_1hop_nei_type == 'item':
                            node_1hop_nei_doc = GraphLoader.item_docs[node_id - 1 - self.user_num]
                        elif node_1hop_nei_type == 'user':
                            node_1hop_nei_doc = GraphLoader.user_docs[node_id - 1]

                        degree = len(node_1hop_nei_doc['hist_%d'%(time_slice)])
                        if degree > 1:
                            node_2hop_candi += node_1hop_nei_doc['hist_%d'%(time_slice)]
                            p_distri += [1/(degree - 1)] * degree

                    if node_2hop_candi != []:
                        p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                        node_2hop_list = np.random.choice(node_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                        node_2hop_t = []
                        for node_2hop_id in node_2hop_list:
                            if node_2hop_nei_feat_dict != None:
                                node_2hop_t.append([node_2hop_id] + node_2hop_nei_feat_dict[str(node_2hop_id)])
                            else:
                                node_2hop_t.append([node_2hop_id])
                        # return node_1hop_t, node_2hop_t
                        self.node_1hop[time_slice] = node_1hop_t
                        self.node_2hop[time_slice] = node_2hop_t
                        with self.work_cnt.get_lock():
                            self.work_cnt.value += 1
                    else:
                        # return node_1hop_t, node_2hop_dummy
                        self.node_1hop[time_slice] = node_1hop_t
                        self.node_2hop[time_slice] = node_2hop_dummy
                        with self.work_cnt.get_lock():
                            self.work_cnt.value += 1
            if self.complete.value == 1:
                return
            
    def gen_user_history(self, start_uid):
        while True:
            if self.work_q.empty() and self.worker_begin.value == 0:
                for i in range(self.time_slice_num):
                    self.work_q.put((start_uid, 'user', i))
                with self.worker_begin.get_lock():
                    self.worker_begin.value = 1
            if self.work_q.empty() and self.worker_begin.value == 1 and self.work_cnt.value == self.time_slice_num:
                user_1hop, user_2hop = self.node_1hop, self.node_2hop
                self.node_1hop = [None] * self.time_slice_num
                self.node_2hop = [None] * self.time_slice_num
                with self.worker_begin.get_lock():
                    self.worker_begin.value = 0
                with self.work_cnt.get_lock():
                    self.work_cnt.value = 0
                return user_1hop, user_2hop

    def gen_item_history(self, start_iid):
        while True:
            if self.work_q.empty() and self.worker_begin.value == 0:
                for i in range(self.time_slice_num):
                    self.work_q.put((start_iid, 'item', i))
                with self.worker_begin.get_lock():
                    self.worker_begin.value = 1
            if self.work_q.empty() and self.worker_begin.value == 1 and self.work_cnt.value == self.time_slice_num:
                item_1hop, item_2hop = self.node_1hop, self.node_2hop
                self.node_1hop = [None] * self.time_slice_num
                self.node_2hop = [None] * self.time_slice_num
                with self.worker_begin.get_lock():
                    self.worker_begin.value = 0
                with self.work_cnt.get_lock():
                    self.work_cnt.value = 0
                return item_1hop, item_2hop

    # def gen_user_history(self, start_uid):
    #     user_1hop, user_2hop = [], []
    #     # t = time.time()
    #     for i in range(self.pred_time):
    #         user_1hop_t, user_2hop_t = self.gen_node_neighbor(start_uid, 'user', i)
    #         user_1hop.append(user_1hop_t)
    #         user_2hop.append(user_2hop_t)
    #     # print('gen_user_history time: {}'.format(time.time() - t))
    #     return user_1hop, user_2hop

    # def gen_item_history(self, start_iid):
    #     item_1hop, item_2hop = [], []
    #     # t = time.time()
    #     for i in range(self.pred_time):
            
    #         item_1hop_t, item_2hop_t = self.gen_node_neighbor(start_iid, 'item', i)
    #         item_1hop.append(item_1hop_t)
    #         item_2hop.append(item_2hop_t)
    #     # print('gen_item_history time: {}'.format(time.time() - t))
    #     return item_1hop, item_2hop
        
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
        curr_uid = 0

        for b in range(line_num):
            line = self.target_f.readline()
            if line == '':
                with self.complete.get_lock():
                    self.complete.value = 1
                raise StopIteration
            line_list = line[:-1].split(',')
            uid = int(line_list[0])
            user_1hop, user_2hop = self.gen_user_history(uid)
            user_1hop_batch += [user_1hop for i in range(1 + NEG_SAMPLE_NUM)]
            user_2hop_batch += [user_2hop for i in range(1 + NEG_SAMPLE_NUM)]
            target_user_batch += [uid] * 10
            if curr_uid != uid:
                for i in range(1 + NEG_SAMPLE_NUM):
                    iid = int(line_list[1 + i])
                    item_1hop, item_2hop = self.gen_item_history(iid)
                    item_1hop_batch.append(item_1hop)
                    item_2hop_batch.append(item_2hop)
                    target_item_batch.append(iid)
                    if i == 0:
                        label_batch.append(1)
                    else:
                        label_batch.append(0)
            else:
                iid = int(line_list[1])
                item_1hop, item_2hop = self.gen_item_history(iid)
                item_1hop_batch.append(item_1hop)
                item_2hop_batch.append(item_2hop)
                item_1hop_batch += item_1hop_batch[-(1 + NEG_SAMPLE_NUM):-1]
                item_2hop_batch += item_2hop_batch[-(1 + NEG_SAMPLE_NUM):-1]
                target_user_batch += [uid] * 10
                target_item_batch.append(iid)
                target_item_batch += target_item_batch[-(1 + NEG_SAMPLE_NUM):-1]
                label_batch += label_batch[-(1 + NEG_SAMPLE_NUM):]

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
    st = time.time()
    i = 0
    for batch_data in graph_loader:
        # print(batch_data[-3:])
        print('batch time: {}'.format(time.time() - t))
        t = time.time()
        i += 1
        if i == 100:
            break
            # print('average time:{}'.format((time.time() - st)/100))
