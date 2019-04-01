import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing

NEG_SAMPLE_NUM = 9
WORKER_N = 5
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
START_TIME = 30

# CCMR dataset parameters
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129

USER_PER_COLLECTION = 1000
ITEM_PER_COLLECTION = 100

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

class GraphHandler(object):
    def __init__(self, time_slice_num, db_name, obj_per_time_slice,
                 user_fnum, item_fnum, user_feat_dict_file = None, 
                 item_feat_dict_file = None):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.user_num = USER_NUM_CCMR
        self.item_num = ITEM_NUM_CCMR
        
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
        
        user_coll_num = self.user_num // USER_PER_COLLECTION
        if self.user_num % USER_PER_COLLECTION != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // ITEM_PER_COLLECTION
        if self.item_num % ITEM_PER_COLLECTION != 0:
            item_coll_num += 1

        self.user_colls = [self.db['user_%d'%(i)] for i in range(user_coll_num)]
        self.item_colls = [self.db['item_%d'%(i)] for i in range(item_coll_num)]

    def gen_node_neighbor(self, start_node_doc, node_type, time_slice):
        if node_type == 'user':
            node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()
            node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
            
            node_1hop_nei_type = 'item'
            node_1hop_nei_fnum = self.item_fnum
            node_1hop_nei_feat_dict = self.item_feat_dict
            node_2hop_nei_feat_dict = self.user_feat_dict

        elif node_type == 'item':
            node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
            node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()

            node_1hop_nei_type = 'user'
            node_1hop_nei_fnum = self.user_fnum
            node_1hop_nei_feat_dict = self.user_feat_dict
            node_2hop_nei_feat_dict = self.item_feat_dict
        
        node_1hop_list = start_node_doc['1hop'][time_slice]
        node_2hop_list = start_node_doc['2hop'][time_slice]
        degree_list = start_node_doc['degrees'][time_slice]
        
        # gen node 2 hops history
        if node_1hop_list == []:
            return node_1hop_dummy, node_2hop_dummy
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
            node_2hop_candi = node_2hop_list
            p_distri = (1 / (np.array(degree_list) - 1)).tolist()
            if node_2hop_candi != []:
                p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                node_2hop_list_choice= np.random.choice(node_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                node_2hop_t = []
                for node_2hop_id in node_2hop_list_choice:
                    if node_2hop_nei_feat_dict != None:
                        node_2hop_t.append([node_2hop_id] + node_2hop_nei_feat_dict[str(node_2hop_id)])
                    else:
                        node_2hop_t.append([node_2hop_id])
                return node_1hop_t, node_2hop_t
            else:
                return node_1hop_t, node_2hop_dummy


    def gen_user_history(self, start_uid, pred_time):
        user_1hop, user_2hop = [], []
        # t = time.time()
        start_node_doc = self.user_colls[(start_uid - 1) // USER_PER_COLLECTION].find({'uid': start_uid})[0]
        for i in range(START_TIME, pred_time):
            user_1hop_t, user_2hop_t = self.gen_node_neighbor(start_node_doc, 'user', i)
            user_1hop.append(user_1hop_t)
            user_2hop.append(user_2hop_t)
        for i in range(TIME_SLICE_NUM_CCMR - pred_time - 1):
            user_1hop.append(user_1hop[-1])
            user_2hop.append(user_2hop[-1])
        # print('gen_user_history time: {}'.format(time.time() - t))
        return user_1hop, user_2hop

    def gen_item_history(self, start_iid, pred_time):
        item_1hop, item_2hop = [], []
        # t = time.time()
        start_node_doc = self.item_colls[(start_iid - self.user_num - 1) // ITEM_PER_COLLECTION].find({'iid':start_iid})[0]
        for i in range(START_TIME, pred_time):
            item_1hop_t, item_2hop_t = self.gen_node_neighbor(start_node_doc, 'item', i)
            item_1hop.append(item_1hop_t)
            item_2hop.append(item_2hop_t)
        for i in range(TIME_SLICE_NUM_CCMR - pred_time - 1):
            item_1hop.append(item_1hop[-1])
            item_2hop.append(item_2hop[-1])
        # print('gen_item_history time: {}'.format(time.time() - t))
        return item_1hop, item_2hop


class GraphLoader(object):
    def __init__(self, graph_handler_params, batch_size, target_file, pred_time, 
                user_feat_dict_file, item_feat_dict_file, worker_n = WORKER_N, 
                max_q_size = 10, wait_time = 0.05):
        self.batch_size = batch_size
        self.max_q_size = max_q_size
        self.wait_time = wait_time
        self.worker_n = worker_n
        self.pred_time = pred_time

        if self.batch_size % 10 != 0:
            print('batch size should be time of {}'.format(1 + NEG_SAMPLE_NUM))
            exit(1)
        self.batch_size2line_num = int(self.batch_size / 10)
        with open(target_file, 'r') as f:
            self.target_lines = f.readlines()
        self.num_of_batch = len(self.target_lines) // self.batch_size2line_num
        if self.num_of_batch * self.batch_size2line_num < len(self.target_lines):
            self.num_of_batch += 1

        # side information dict
        self.user_feat_dict = None
        self.item_feat_dict = None
        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        if item_feat_dict_file != None:
            with open(item_feat_dict_file, 'rb') as f:
                self.item_feat_dict = pkl.load(f)

        # multiprocessing
        self.prod_batch_num = 0 # for producer
        self.work = multiprocessing.Queue(maxsize=self.max_q_size)
        self.results = multiprocessing.Queue(maxsize=self.max_q_size)
        self.producer_stop = multiprocessing.Value('d', 0)
        self.worker_stop = multiprocessing.Value('d', 0)
        self.threads = []

        thread = multiprocessing.Process(target=self.producer)
        self.threads.append(thread)
        thread.daemon = True
        thread.start()
        for i in range(worker_n):
            thread = multiprocessing.Process(target=self.worker, args=[graph_handler_params])
            self.threads.append(thread)
            thread.daemon = True
            thread.start()
    
    def producer(self):
        while self.producer_stop.value == 0:
            uids = []
            iids = []
            if (self.prod_batch_num + 1) * self.batch_size2line_num <= len(self.target_lines):
                lines = self.target_lines[self.prod_batch_num * self.batch_size2line_num : (self.prod_batch_num + 1) * self.batch_size2line_num]
            else:
                lines = self.target_lines[self.prod_batch_num * self.batch_size2line_num :]
            self.prod_batch_num += 1
            for line in lines:
                line_list = line[:-1].split(',')
                uids.append(line_list[0])
                iids += line_list[1:]
            uids = [int(uid) for uid in uids]
            iids = [int(iid) for iid in iids]
            while self.work.qsize() >= self.max_q_size:
                time.sleep(self.wait_time)
            self.work.put((uids, iids))
            if self.prod_batch_num == self.num_of_batch:
                with self.producer_stop.get_lock():
                    self.producer_stop.value = 1
                    break
    
    def worker(self, params):
        graph_handler = GraphHandler(params[0], params[1], params[2], params[3], params[4], params[5], params[6])

        while not (self.work.qsize() == 0 and self.producer_stop.value == 1):
            try:
                uids, iids = self.work.get(timeout=self.wait_time)
            except:
                continue
            user_1hop_batch = []
            user_2hop_batch = []
            item_1hop_batch = []
            item_2hop_batch = []
            target_user_batch = []
            target_item_batch = []
            label_batch = []
            length_batch = []

            for i in range(len(uids)):
                user_1hop, user_2hop = graph_handler.gen_user_history(uids[i], self.pred_time)
                for j in range(i * (NEG_SAMPLE_NUM + 1), (i + 1) * (NEG_SAMPLE_NUM + 1)):
                    item_1hop, item_2hop = graph_handler.gen_item_history(iids[j], self.pred_time)
                    user_1hop_batch.append(user_1hop)
                    user_2hop_batch.append(user_2hop)
                    item_1hop_batch.append(item_1hop)
                    item_2hop_batch.append(item_2hop)
                    if self.user_feat_dict == None:
                        target_user_batch.append([uids[i]])
                    else:
                        target_user_batch.append([uids[i]] + self.user_feat_dict[str(uids[i])])
                    if self.item_feat_dict == None:
                        target_item_batch.append([iids[j]])
                    else:
                        target_item_batch.append([iids[j]] + self.item_feat_dict[str(iids[j])])
                    if j % (NEG_SAMPLE_NUM + 1) == 0:
                        label_batch.append(1)
                    else:
                        label_batch.append(0)
                    length_batch.append(self.pred_time - START_TIME)
            self.results.put((user_1hop_batch, user_2hop_batch, item_1hop_batch, item_2hop_batch, target_user_batch, target_item_batch, label_batch, length_batch))
        with self.worker_stop.get_lock():
            self.worker_stop.value += 1

    def __iter__(self):
        return self

    def __next__(self):
        while self.results.empty() and self.worker_stop.value != self.worker_n:
            time.sleep(self.wait_time)
        if self.results.empty() and self.worker_stop.value == self.worker_n:
            for thread in self.threads:
                thread.terminate()
            raise StopIteration
        re = self.results.get()
        return re

if __name__ == "__main__":
    graph_handler_params = [TIME_SLICE_NUM_CCMR, 'ccmr_2hop', OBJ_PER_TIME_SLICE_CCMR, \
                            1, 5, None, DATA_DIR_CCMR + 'remap_movie_info_dict.pkl']
    # graph_handler = GraphHandler(TIME_SLICE_NUM_CCMR,
    #                             'ccmr_2hop',
    #                             OBJ_PER_TIME_SLICE_CCMR,
    #                             1,
    #                             5,
    #                             None, 
    #                             DATA_DIR_CCMR + 'remap_movie_info_dict.pkl')
    # for i in range(1, 100):
    #     graph_handler.gen_user_history(i, 40)
    # for i in range(USER_NUM_CCMR + 1 + 10, USER_NUM_CCMR + 1 + 100):
    #     graph_handler.gen_item_history(i, 40)
    graph_loader = GraphLoader(graph_handler_params, 100, DATA_DIR_CCMR + 'target_train.txt', 39, None, DATA_DIR_CCMR + 'remap_movie_info_dict.pkl')
    
    t = time.time()
    st = time.time()
    i = 1
    for batch_data in graph_loader:
        print('batch time of batch-{}: {}'.format(i, (time.time() - t)))
        i += 1
        t = time.time()
    print('total time:{}'.format(time.time() - st))
