import pymongo
import pickle as pkl
import datetime
import time

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
        # construct user collection
        user_doc_list = []
        for i in range(1, self.user_num + 1):
            user_doc_list.append(self.gen_user_doc(i))
        print('user doc list completed')
        self.user_coll.insert_many(user_doc_list)
        print('user collection completed')

        # construct item collection
        item_doc_list = []
        for i in range(self.user_num + 1, self.user_num + self.item_num + 1):
            item_doc_list.append(self.gen_item_doc(i))
        print('item doc list completed')
        self.item_coll.insert_many(item_doc_list)
        print('item collection completed')

    def write2db():
        for line in self.rating_file:
            uid, iid, _, t_idx = line[:-1].split(',')
            user_doc = self.user.coll.find_one({'uid': int(uid)})
            user_doc['hist_%s'%(t_idx)].append(int(iid))
            item_doc = self.item.coll.find_one({'iid': int(iid)})
            item_doc['hist_%s'%(t_idx)].append(int(uid))
        print('write to db complete')


if __name__ == "__main__":
    # For CCMR
    gs = CCMRGraphStore(DATA_DIR + 'remap_rating_pos.csv', DATA_DIR + 'remap_movie_info_dict.pkl')
    gs.construct_coll()
    gs.write2db()