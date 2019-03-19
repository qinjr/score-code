import pymongo
import pickle as pkl
import datetime
import time

SECONDS_PER_DAY = 24 * 3600

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
        self.rating_file = open(rating_file, 'r')
        with open(movie_info_dict, 'rb') as f:
            self.movie_info_dict = pkl.load(f)
        
    
    def date2idx(self, time_str):
        time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
        return

    def write2db():
        for line in self.in_file:
            uid, iid, _, time = line[:-1].split(,)
