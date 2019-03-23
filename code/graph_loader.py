import random
import pymongo
import pickle as pkl
import time

NEG_SAMPLE_NUM = 9
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'

# CCMR dataset parameters
USER_SIDE_FEATURE_CCMR = []
ITEM_SIDE_FEATURE_CCMR = ['did', 'aid', 'gid', 'nid']
TIME_SLICE_NUM_CCMR = 41
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129

class GraphLoader(object):
    def __init__(self, time_slice_num, db_name, user_neg_dict_file):
        self.url = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(self.url)
        self.db = self.client[db_name]
        self.user_coll = self.db.user
        self.item_coll = self.db.item
        self.user_num = self.user_coll.find().count()
        self.item_num = self.item_coll.find().count()

        with open(user_neg_dict_file, 'rb') as f:
            self.user_neg_dict = pkl.load(f)
        self.time_slice_num = time_slice_num
    
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


if __name__ == "__main__":
    graph_loader = GraphLoader(TIME_SLICE_NUM_CCMR, 'ccmr', DATA_DIR_CCMR + 'user_neg_dict.pkl')
    graph_loader.gen_target_file(TIME_SLICE_NUM_CCMR - 2, NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_train.txt')
    graph_loader.gen_target_file(TIME_SLICE_NUM_CCMR - 1, NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_test.txt')
