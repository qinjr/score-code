import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing

NEG_SAMPLE_NUM = 9
MAX_LEN = 80
WORKER_N = 10
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
START_TIME = 30

# CCMR dataset parameters
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129

USER_PER_COLLECTION = 1000
ITEM_PER_COLLECTION = 1000

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

class GraphLoader(object):
    def __init__(self, time_slice_num, db_name, obj_per_time_slice, target_file,
                 user_fnum, item_fnum, user_feat_dict_file, item_feat_dict_file,
                 batch_size, pred_time, worker_n=WORKER_N, wait_time=0.001):
        self.db_name = db_name
        self.user_num = USER_NUM_CCMR
        self.item_num = ITEM_NUM_CCMR
        self.obj_per_time_slice = obj_per_time_slice
        self.time_slice_num = time_slice_num

        self.user_fnum = user_fnum
        self.item_fnum = item_fnum
        self.user_feat_dict = None
        self.item_feat_dict = None
        
        self.target_f = open(target_file, 'r')
        # side information dict
        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        if item_feat_dict_file != None:
            with open(item_feat_dict_file, 'rb') as f:
                self.item_feat_dict = pkl.load(f)
        print('graph loader initial completed')

        self.batch_size = batch_size
        self.pred_time = pred_time

        # multiprocessing
        self.worker_n = worker_n
        self.wait_time = wait_time

        self.processes = []
        self.work_q = multiprocessing.Queue()
        self.result_1hop_q = multiprocessing.Queue()
        self.result_2hop_q = multiprocessing.Queue()
        self.work_cnt = multiprocessing.Value('d', self.pred_time - START_TIME)
        self.work_begin = multiprocessing.Event()
        self.work_end = multiprocessing.Event()
        self.complete = multiprocessing.Value('d', 0)

        for i in range(worker_n):
            process = multiprocessing.Process(target=self.gen_node_neighbor, args=[i])
            self.processes.append(process)
            process.daemon = True
            process.start()
    
    def gen_node_neighbor(self, name):
        url = "mongodb://localhost:27017/"
        client = pymongo.MongoClient(url)
        db = client[self.db_name]

        user_coll_num = self.user_num // USER_PER_COLLECTION
        if self.user_num % USER_PER_COLLECTION != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // ITEM_PER_COLLECTION
        if self.item_num % ITEM_PER_COLLECTION != 0:
            item_coll_num += 1

        user_colls = [db['user_%d'%(i)] for i in range(user_coll_num)]
        item_colls = [db['item_%d'%(i)] for i in range(item_coll_num)]
        
        while True:
            if self.complete.value == 1:
                return
            self.work_begin.wait()
            # if self.work_cnt.value == self.pred_time - START_TIME:
            #     time.sleep(self.wait_time)
            # else:
            try:
                start_node_id, node_type, time_slice = self.work_q.get(timeout=self.wait_time)
            except:
                continue
            # t=time.time()
            if node_type == 'user':
                # start_node_doc = self.user_coll.find({'uid': start_node_id})[0]
                start_node_doc = user_colls[(start_node_id - 1) // USER_PER_COLLECTION].find({'uid': start_node_id})[0]#user_cursor[start_node_id - 1]
                node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()
                node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
                
                node_1hop_nei_type = 'item'
                node_1hop_nei_fnum = self.item_fnum
                node_1hop_nei_feat_dict = self.item_feat_dict
                node_2hop_nei_feat_dict = self.user_feat_dict

            elif node_type == 'item':
                # start_node_doc = self.item_coll.find({'iid': start_node_id})[0]
                start_node_doc = item_colls[(start_node_id - self.user_num - 1) // ITEM_PER_COLLECTION].find({'iid':start_node_id})[0]#item_cursor[start_node_id - 1 - self.user_num]
                node_1hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.user_fnum), dtype=np.int).tolist()
                node_2hop_dummy = np.zeros(shape=(self.obj_per_time_slice, self.item_fnum), dtype=np.int).tolist()

                node_1hop_nei_type = 'user'
                node_1hop_nei_fnum = self.user_fnum
                node_1hop_nei_feat_dict = self.user_feat_dict
                node_2hop_nei_feat_dict = self.item_feat_dict
            
            # node_1hop_list = start_node_doc['hist_%d'%(time_slice)] #[iid1, iid2, ...]
            node_1hop_list = start_node_doc['1hop'][time_slice] #[iid1, iid2, ...]
            node_2hop_list = start_node_doc['2hop'][time_slice]
            degree_list = start_node_doc['degrees'][time_slice]
            # print('phase1 time: {}'.format(time.time()-t))
            
            # gen node 2 hops history
            if node_1hop_list == []:
                self.result_1hop_q.put((node_1hop_dummy, time_slice))
                self.result_2hop_q.put((node_2hop_dummy, time_slice))
                with self.work_cnt.get_lock():
                    # print('worker time: {}'.format(time.time()-t))
                    self.work_cnt.value += 1
                    if self.work_cnt.value == self.pred_time - START_TIME:
                        self.work_begin.clear()
                        self.work_end.set()
                # return node_1hop_dummy, node_2hop_dummy
            else:
                # t=time.time()
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
                # print('phase2 time: {}'.format(time.time()-t))
                # st=time.time()
                # deal with 2hop            
                node_2hop_candi = node_2hop_list#[]
                p_distri = (1 / (np.array(degree_list) - 1)).tolist()#[]
                # for node_id in node_1hop_list_unique:
                #     if node_1hop_nei_type == 'item':
                #         # t=time.time()
                #         node_1hop_nei_doc = item_colls[(node_id - self.user_num - 1) // ITEM_PER_COLLECTION].find({'iid':node_id})[0]#item_cursor[node_id - 1 - self.user_num]
                #         # print('find item time: {}'.format(time.time()-t))
                #         # node_1hop_nei_doc = self.item_coll.find_one({'iid': node_id})
                #     elif node_1hop_nei_type == 'user':
                #         # t=time.time()
                #         node_1hop_nei_doc = user_colls[(node_id - 1) // USER_PER_COLLECTION].find({'uid': node_id})[0]#user_cursor[node_id - 1]
                #         # print('find user time: {}'.format(time.time()-t))
                #         # node_1hop_nei_doc = self.user_coll.find_one({'uid': node_id})
                #     content = node_1hop_nei_doc['hist_%d'%(time_slice)]
                #     degree = len(content)
                #     if degree > 1:
                #         node_2hop_candi += content
                #         p_distri += [1/(degree - 1)] * degree
                # print('phase3 time: {}'.format(time.time()-st))
                # t=time.time()
                if node_2hop_candi != []:
                    p_distri = (np.exp(p_distri) / np.sum(np.exp(p_distri))).tolist()
                    node_2hop_list_choice= np.random.choice(node_2hop_candi, self.obj_per_time_slice, p=p_distri).tolist()
                    node_2hop_t = []
                    for node_2hop_id in node_2hop_list_choice:
                        if node_2hop_nei_feat_dict != None:
                            node_2hop_t.append([node_2hop_id] + node_2hop_nei_feat_dict[str(node_2hop_id)])
                        else:
                            node_2hop_t.append([node_2hop_id])
                    # print('phase4 time: {}'.format(time.time()-t))
                    self.result_1hop_q.put((node_1hop_t, time_slice))
                    self.result_2hop_q.put((node_2hop_t, time_slice))
                    with self.work_cnt.get_lock():
                        # print('phase 4 time: {}'.format(time.time()-t))
                        # print('worker time: {}'.format(time.time()-t))
                        self.work_cnt.value += 1
                        if self.work_cnt.value == self.pred_time - START_TIME:
                            self.work_begin.clear()
                            self.work_end.set()
                    # return node_1hop_t, node_2hop_t
                else:
                    self.result_1hop_q.put((node_1hop_t, time_slice))
                    self.result_2hop_q.put((node_2hop_dummy, time_slice))
                    with self.work_cnt.get_lock():
                        # print('phase 4 time: {}'.format(time.time()-t))
                        # print('worker time: {}'.format(time.time()-t))
                        self.work_cnt.value += 1
                        if self.work_cnt.value == self.pred_time - START_TIME:
                            self.work_begin.clear()
                            self.work_end.set()
                    # return node_1hop_t, node_2hop_dummy

    def gen_user_history(self, start_uid):
        for i in range(START_TIME, self.pred_time):
            self.work_q.put((start_uid, 'user', i))
        with self.work_cnt.get_lock():
            self.work_cnt.value = 0
            self.work_end.clear()
            self.work_begin.set()
        # time.sleep(self.wait_time)
        # while True:
        self.work_end.wait()
        # if self.work_cnt.value == self.pred_time - START_TIME:
        user_1hop_list, user_2hop_list = [], []
        user_1hop, user_2hop = [], []
        for i in range(self.pred_time - START_TIME):
            user_1hop_list.append(self.result_1hop_q.get())
            user_2hop_list.append(self.result_2hop_q.get())
        user_1hop_list = sorted(user_1hop_list, key=lambda tup:tup[1])
        user_2hop_list = sorted(user_2hop_list, key=lambda tup:tup[1])
        for i in range(self.pred_time - START_TIME):
            user_1hop.append(user_1hop_list[i][0])
            user_2hop.append(user_2hop_list[i][0])
        return user_1hop, user_2hop
        # else:
        #     time.sleep(self.wait_time)
    
    def gen_item_history(self, start_iid):
        for i in range(START_TIME, self.pred_time):
            self.work_q.put((start_iid, 'item', i))
        with self.work_cnt.get_lock():
            self.work_cnt.value = 0
            self.work_end.clear()
            self.work_begin.set()
        # time.sleep(self.wait_time)
        # while True:
        self.work_end.wait()
        # if self.work_cnt.value == self.pred_time - START_TIME:
        item_1hop_list, item_2hop_list = [], []
        item_1hop, item_2hop = [], []
        for i in range(self.pred_time - START_TIME):
            item_1hop_list.append(self.result_1hop_q.get())
            item_2hop_list.append(self.result_2hop_q.get())
        item_1hop_list = sorted(item_1hop_list, key=lambda tup:tup[1])
        item_2hop_list = sorted(item_2hop_list, key=lambda tup:tup[1])
        for i in range(self.pred_time - START_TIME):
            item_1hop.append(item_1hop_list[i][0])
            item_2hop.append(item_2hop_list[i][0])
        return item_1hop, item_2hop
        # else:
        #     time.sleep(self.wait_time)

    def __iter__(self):
        return self
    
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
            # if curr_uid != uid:
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
            # else:
            #     iid = int(line_list[1])
            #     item_1hop, item_2hop = self.gen_item_history(iid)
            #     item_1hop_batch.append(item_1hop)
            #     item_2hop_batch.append(item_2hop)
            #     item_1hop_batch += item_1hop_batch[-(1 + NEG_SAMPLE_NUM):-1]
            #     item_2hop_batch += item_2hop_batch[-(1 + NEG_SAMPLE_NUM):-1]
            #     target_user_batch += [uid] * 10
            #     target_item_batch.append(iid)
            #     target_item_batch += target_item_batch[-(1 + NEG_SAMPLE_NUM):-1]
            #     label_batch += label_batch[-(1 + NEG_SAMPLE_NUM):]

        return [user_1hop_batch, user_2hop_batch, item_1hop_batch, item_2hop_batch, target_user_batch, target_item_batch, label_batch]

if __name__ == "__main__":
    graph_loader = GraphLoader(TIME_SLICE_NUM_CCMR, 
                                'ccmr_2hop', 
                                OBJ_PER_TIME_SLICE_CCMR,
                                DATA_DIR_CCMR + 'target_train.txt',
                                1,
                                5,
                                None,
                                DATA_DIR_CCMR + 'remap_movie_info_dict.pkl', 
                                100, 
                                39)

    # for i in range(400, 450):
    #     t = time.time()
    #     user_1hop, user_2hop = graph_loader.gen_user_history(i)
    #     print('user gen time: {}'.format(time.time() - t))
    
    # for i in range(1 + USER_NUM_CCMR + 500, 1 + USER_NUM_CCMR + 550):
    #     t = time.time()
    #     item_1hop, item_2hop = graph_loader.gen_item_history(i)
    #     print('item gen time: {}'.format(time.time() - t))
    
    t = time.time()
    st = time.time()
    for batch_data in graph_loader:
        print('batch_time: {}'.format(time.time() - t))
        t = time.time()
    print('total time: {}'.format(time.time() - st))


    # tg = TargetGen(DATA_DIR_CCMR + 'user_neg_dict.pkl', 'ccmr_1hop')
    # tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_train.txt', 39)
    # tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_test.txt', 40)