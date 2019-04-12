import matplotlib
matplotlib.use('Agg')
import pickle as pkl
import datetime
import time
import matplotlib.pyplot as plt

RAW_DIR = '../../score-data/Tmall/raw_data/'
FEATENG_DIR = '../../score-data/Tmall/feateng/'

def join_user_profile(user_profile_file, behavior_file, joined_file):
    user_profile_dict = {}
    with open(user_profile_file, 'r') as f:
        for line in f:
            uid, aid, gid = line[:-1].split(',')
            user_profile_dict[uid] = ','.join([aid, gid])
    
    # join
    newlines = []
    with open(behavior_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            user_profile = user_profile_dict[uid]
            newlines.append(line[:-1] + ',' + user_profile + '\n')
    with open(joined_file, 'w') as f:
        f.writelines(newlines)
    
if __name__ == "__main__":
    join_user_profile(RAW_DIR + 'user_info_format1.csv', RAW_DIR + 'user_log_format1.csv', RAW_DIR + 'joined_user_behavior.csv')
