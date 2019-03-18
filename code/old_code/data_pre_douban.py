from collections import Counter
import cPickle as pkl
import random
import datetime
import time
import pandas as pd
random.seed(11)

K = 5

def cal_item_popularity(inFile = '../../data/douban/rating_logs_f.csv', outFile = '../../data/douban/top_item_list.pkl'):
    item_frequency = {}
    ptr1 = 0
    ptr2 = 0
    cur_iid = '135169'
    with open(inFile, 'r') as f:
        for line in f:
            line_iid = line.split(',')[1]
            ptr1 += 1
            if cur_iid != line_iid:
                item_frequency[cur_iid] = ptr1 - ptr2
                cur_iid = line_iid
                ptr2 = ptr1
        item_frequency[cur_iid] = ptr1 - ptr2 - 1
        cur_iid = line_iid
        ptr2 = ptr1
    # find TOP items
    d = Counter(item_frequency)
    top_item_tup = d.most_common()
    top_item_list = []
    sample_cnt = 0
    for i in range(5000):
        top_item_list.append(top_item_tup[i][0])
        sample_cnt += item_frequency[top_item_tup[i][0]]
    with open(outFile, 'w') as f:
        pkl.dump(top_item_list, f)
    print("sample_cnt: {}".format(sample_cnt))

def cal_user_iteraction(inFile = '../../data/douban/rating_logs_f.csv', outFile = '../../data/douban/top_user_list.pkl'):
    user_iteraction = {}

    with open(inFile, 'r') as f:
        for line in f:
            line_uid = line.split(',')[0]
            if line_uid in user_iteraction:
                user_iteraction[line_uid] += 1
            else:
                user_iteraction[line_uid] = 1

    # find TOP users
    d = Counter(user_iteraction)
    top_user_tup = d.most_common()
    top_user_list = []
    sample_cnt = 0
    for i in range(1000000):
        top_user_list.append(top_user_tup[i][0])
        sample_cnt += user_iteraction[top_user_tup[i][0]]
    with open(outFile, 'w') as f:
        pkl.dump(top_user_list, f)
    print("sample_cnt: {}".format(sample_cnt))

def sample_top_user(inFile = '../../data/douban/rating_logs_f.csv', top_user_list_file = '../../data/douban/top_user_list.pkl',
                     outFile = '../../data/douban/rating_logs_top.csv'):
    with open(top_user_list_file, 'r') as f:
        top_user_list = pkl.load(f)
        top_user_dict = dict(zip(top_user_list, range(len(top_user_list))))

    newlines = []
    with open(inFile, 'r') as f:
        for line in f:
            uid = line.split(',')[0]
            if uid in top_user_dict:
                newlines.append(line)
    
    with open(outFile, 'w') as f:
        f.writelines(newlines)

def sample_recent5year(inFile = '../../data/douban/rating_logs_top.csv', outFile = '../../data/douban/rating_logs_score.csv'):
    newlines = []
    with open(inFile, 'r') as f:
        for line in f:
            line_items = line[:-1].split(',')
            if line_items[-1][:4] >= '2012':
                newlines.append(line)
    with open(outFile, 'w') as f:
        f.writelines(newlines)

def get_movie_info(inFile = '../../data/douban/movie_info.csv', outFile = '../../data/douban/movie_cate_dict.pkl'):
    movie_cate_dict = {}
    with open(inFile, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            iid, cid = line[:-1].split(',')[0], line[:-1].split(',')[-3].split(';')[0]
            movie_cate_dict[iid] = cid
    
    with open(outFile, 'wb') as f:
        pkl.dump(movie_cate_dict, f)

def aggregate(inFile = '../../data/douban/rating_logs_score.csv', movie_cate_dict_file = 'movie_cate_dict.pkl', 
                        outFile = '../../data/douban/rating_logs_score_complete.csv'):
    with open(movie_cate_dict_file, 'r') as f:
        movie_cate_dict = pkl.load(f)
    
    newlines = []
    with open(inFile, 'r') as f:
        for line in f:
            line_item = line[:-1].split(',')
            cid = movie_cate_dict[line_item[1]]
            newlines.append(','.join(line_item[:2] + [cid] + line_item[2:]) + '\n')
    
    with open(outFile, 'w') as f:
        f.writelines(newlines)

def time2slice(inFile = '../../data/douban/rating_logs_score_complete.csv', outFile = '../../data/douban/rating_logs_score_complete.csv'):
    newlines = []
    start_time = int(time.mktime(datetime.datetime.strptime('2012-01-01', "%Y-%m-%d").timetuple()))
    time_interval = 24 * 3600 * 60
    
    max_tid = 0
    min_tid = 50
    with open(inFile, 'r') as f:
        for line in f:
            line_items = line[:-1].split(',')
            tstamp = int(time.mktime(datetime.datetime.strptime(line_items[-1], "%Y-%m-%d").timetuple()))
            tid = (tstamp - start_time) / time_interval
            if tid > max_tid:
                max_tid = tid
            if tid < min_tid:
                min_tid = tid
            newlines.append(','.join(line_items[:-1] + [str(tid)]) + '\n')
    
    with open(outFile, 'w') as f:
        f.writelines(newlines)

    print('max tid: {}'.format(max_tid))
    print('min tid: {}'.format(min_tid))

def gen_label(inFile = '../../data/douban/rating_logs_score_complete.csv', outFile = '../../data/douban/rating_logs_score_complete.csv'):
    newlines = []
    with open(inFile, 'r') as f:
        for line in f:
            line_items = line[:-1].split(',')
            if line_items[3] == '0' or line_items[3] == '4':
                continue
            elif line_items[3] == '5':
                line_items[3] = '1'
            else:
                line_items[3] = '0'
            newlines.append(','.join(line_items) + '\n')
    
    with open(outFile, 'w') as f:
        f.writelines(newlines)

def remap(inFile = '../../data/douban/rating_logs_score_complete.csv', outFile = '../../data/douban/rating_logs_score_complete.csv'):
    df = pd.read_csv(inFile, names=['uid', 'iid', 'cid', 'label', 'time'])
    df['cid'].fillna(0, inplace=True)

    uid_key = sorted(df['uid'].unique().tolist())
    uid_num = len(uid_key)
    uid_map = dict(zip(uid_key, range(uid_num)))
    df['uid'] = df['uid'].map(lambda x: uid_map[x])

    iid_key = sorted(df['iid'].unique().tolist())
    iid_num = len(iid_key)
    iid_map = dict(zip(iid_key, range(uid_num, iid_num+uid_num)))
    df['iid'] = df['iid'].map(lambda x: iid_map[x])

    cid_key = sorted(df['cid'].unique().tolist())
    cid_num = len(cid_key)
    cid_map = dict(zip(cid_key, range(iid_num+uid_num, iid_num+uid_num+cid_num)))
    df['cid'] = df['cid'].map(lambda x: cid_map[x])

    print('feature size: {}'.format(iid_num+uid_num+cid_num))
    df = df.sort_values(by=['uid', 'time'])

    df.to_csv(outFile, header=False, index=False)

def split_pos_neg(inFile = '../../data/douban/rating_logs_score_complete.csv', outFile = '../../data/douban/rating_logs_final.csv'):
    df = pd.read_csv(inFile, names=['uid', 'iid', 'cid', 'label', 'time'])
    df = df.reindex(columns = ['uid','iid','cid','time','label'])
    df_pos = df.loc[df['label'] == 1]
    df_neg = df.loc[df['label'] == 0]

    item_list_pos = df_pos['iid'].unique().tolist()
    item_list_pos_dict = dict(zip(item_list_pos, range(len(item_list_pos))))
    item_cate_dict = dict(zip(df['iid'].tolist(), df['cid'].tolist()))

    user_neg_dict = {}
    for uid, hist in df_neg.groupby('uid'):
        uid = str(uid)
        user_neg_dict[uid] = []
        for iid in hist['iid'].tolist():
            if iid in item_list_pos_dict:
                user_neg_dict[uid].append([str(iid), str(item_cate_dict[iid])])
        if len(user_neg_dict[uid]) >= 4:
            user_neg_dict[uid] = user_neg_dict[uid][:4]
        else:
            for i in range(4 - len(user_neg_dict[uid])):
                rand_idx = random.randint(0, len(item_list_pos) - 1)
                user_neg_dict[uid].append([str(item_list_pos[rand_idx]), str(item_cate_dict[item_list_pos[rand_idx]])])
    
    for uid in df_pos['uid'].unique().tolist():
        uid = str(uid)
        if uid not in user_neg_dict:
            user_neg_dict[uid] = []
            for i in range(4):
                rand_idx = random.randint(0, len(item_list_pos) - 1)
                user_neg_dict[uid].append([str(item_list_pos[rand_idx]), str(item_cate_dict[item_list_pos[rand_idx]])])
    
    df_pos.to_csv(outFile, header=False, index=False)
    with open('../../data/douban/user_neg_dict.pkl', 'w') as f:
        pkl.dump(user_neg_dict, f)
    
    for u in user_neg_dict.keys():
        for tup in user_neg_dict[u]:
            if int(tup[0]) not in item_list_pos_dict:
                print('WRONG')
                exit(1)

    print('split pos neg complete')

def gen_user_item_seq(in_file = '../../data/douban/rating_logs_final.csv', out_user_file = '../../data/douban/score-data/user-side.txt', 
                      out_item_file = '../../data/douban/score-data/item-side.txt'):
    item_side_dict = {}
    lines_user = []
    lines_item = []

    with open(in_file, 'r') as f:
        lines = f.readlines()
        cur_uid = '-1'
        user_line_list = []
        for line in lines:
            line_split = line[:-1].split(',')
            uid, iid, cid, tid = line_split[0], line_split[1], line_split[2], line_split[3]
            # for user-side seq
            if cur_uid != uid:
                # set new cur uid and the last user line list is done
                cur_uid = uid
                if user_line_list != []:
                    for i in range(1, len(user_line_list)):
                        user_line_list[i] = ' '.join(user_line_list[i])
                    user_line = '\t'.join(user_line_list) + '\n'
                    lines_user.append(user_line)
                    user_line_list = []
                # start a new user
                user_line_list.append(uid)
                user_line_list.append([','.join([iid, cid, tid])])
            else:
                if int(tid) > int(user_line_list[-1][0].split(',')[-1]):
                    user_line_list.append([','.join([iid, cid, tid])])
                else:
                    user_line_list[-1].append(','.join([iid, cid, tid]))

            # for item-side seq
            if iid not in item_side_dict:
                item_side_dict[iid] = [(uid, tid)]
            else:
                item_side_dict[iid].append((uid, tid))

        # handle the last user
        if user_line_list != []:
            for i in range(1, len(user_line_list)):
                user_line_list[i] = ' '.join(user_line_list[i])
            user_line = '\t'.join(user_line_list) + '\n'
            lines_user.append(user_line)

        # user size file completed
        with open(out_user_file, 'w') as f:
            f.writelines(lines_user)
        print('user side completed')

        # sort item side seq
        for iid in item_side_dict:
            item_side_dict[iid].sort(key=lambda tup: int(tup[1]))
        print('item seq sort completed')

        # gen item side lines
        for iid in item_side_dict:
            item_line_list = [iid, [','.join(item_side_dict[iid][0])]]
            if len(item_side_dict[iid]) > 1:
                for tup in item_side_dict[iid][1:]:
                    if int(tup[-1]) > int(item_line_list[-1][0].split(',')[-1]):
                        item_line_list.append([','.join(tup)])
                    else:
                        item_line_list[-1].append(','.join(tup))
            
            for i in range(1, len(item_line_list)):
                item_line_list[i] = ' '.join(item_line_list[i])
            item_line = '\t'.join(item_line_list) + '\n'
            lines_item.append(item_line)
            item_line_list = []

        with open(out_item_file, 'w') as f:
            f.writelines(lines_item)
        print('item side completed')

def gen_1hop(user_in_file = '../../data/douban/score-data/user-side.txt', item_in_file = '../../data/douban/score-data/item-side.txt', 
             user_neg_dict_file = '../../data/douban/user_neg_dict.pkl',  user_1hop_file = '../../data/douban/score-data/user-1hop.txt', 
             item_1hop_file = '../../data/douban/score-data/item-1hop.txt', target_item_file = '../../data/douban/score-data/target.txt'):
    user_1hop_lines = []
    item_1hop_lines = []
    target_item_list = []
    item_seq_dict = {}
    
    neg_num = 4
    # read neg list dict, {uid: neg list}
    with open(user_neg_dict_file, 'r') as f:
        user_neg_dict = pkl.load(f)

    # get item seq dict,{iid: item seq str}
    with open(item_in_file) as f:
        lines = f.readlines()
        for line in lines:
            iid, item_seq = line.split('\t')[0], line.replace(line.split('\t')[0] + '\t', '')
            item_seq_dict[iid] = item_seq
        print('construct item seq dict completed')

    with open(user_in_file) as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split('\t')) <= 2:
                continue
            uid = line.split('\t')[0]
            pos_item = line[:-1].split('\t')[-1].split(' ')[-1] #iid,cid,tid
            neg_item_list = user_neg_dict[uid]
            if line[:-1].split('\t')[-1].split(' ')[:-1] != []:
                user_his_seq = '\t'.join(line[:-1].split('\t')[:-1] + [' '.join(line[:-1].split('\t')[-1].split(' ')[:-1])])
            else:
                user_his_seq = '\t'.join(line[:-1].split('\t')[:-1])
            
            user_pos_neg_line_list = []
            item_pos_neg_line_list = []
            target_pos_neg_line_list = []

            for i in range(neg_num + 1):
                if i == 0:
                    target_item = pos_item
                    label = 1
                else:
                    target_item = ','.join([neg_item_list[i-1][0], neg_item_list[i-1][1], pos_item.split(',')[2]])
                    label = 0

                target_item_iid = target_item.split(',')[0]
                target_item_tid = target_item.split(',')[2]

                # time slice sync
                item_his_seq = target_item_iid + '\t' + item_seq_dict[target_item_iid][:-1]
                user_his_seq_list = user_his_seq.split('\t')
                item_his_seq_list = item_his_seq.split('\t')

                start_time = min(int(user_his_seq_list[1].split(' ')[0].split(',')[-1]), int(item_his_seq_list[1].split(' ')[0].split(',')[-1]))
                num_time_slice = int(target_item_tid) - 1 - start_time + 1
                numb_node = '0,0,0'
                sync_user_his_seq_list = [user_his_seq_list[0]] + [numb_node for j in range(num_time_slice)]
                sync_item_his_seq_list = [item_his_seq_list[0]] + [numb_node for j in range(num_time_slice)]

                user_cnt = 0
                item_cnt = 0
                for t in user_his_seq_list[1:]:
                    if int(t.split(' ')[0].split(',')[-1]) < int(target_item_tid):
                        user_cnt += 1
                        if len(t.split(' ')) > K:
                            t = ' '.join(t.split(' ')[:K])
                        sync_user_his_seq_list[int(t.split(' ')[0].split(',')[-1]) - start_time + 1] = t

                for t in item_his_seq_list[1:]:
                    if int(t.split(' ')[0].split(',')[-1]) < int(target_item_tid):
                        item_cnt += 1
                        if len(t.split(' ')) > K:
                            t = ' '.join(t.split(' ')[:K])
                        sync_item_his_seq_list[int(t.split(' ')[0].split(',')[-1]) - start_time + 1] = t
                if user_cnt == 0 or item_cnt == 0:
                    continue #remove too sparse samples

                if sync_user_his_seq_list[-1] == sync_item_his_seq_list[-1]:
                    sync_user_his_seq_list = sync_user_his_seq_list[:-1]
                    sync_item_his_seq_list = sync_item_his_seq_list[:-1]

                user_1hop_line = '\t'.join(sync_user_his_seq_list) + '\n'
                item_1hop_line = '\t'.join(sync_item_his_seq_list) + '\n'

                user_pos_neg_line_list.append(user_1hop_line)
                item_pos_neg_line_list.append(item_1hop_line)
                target_pos_neg_line_list.append(uid + ',' +target_item + ',' + str(label) + '\n') # uid,iid,cid,tid,label

            if len(user_pos_neg_line_list) == neg_num + 1:
                user_1hop_lines += user_pos_neg_line_list
                item_1hop_lines += item_pos_neg_line_list
                target_item_list += target_pos_neg_line_list
            else:
                continue

        # write files
        with open(target_item_file, 'w') as f:
            f.writelines(target_item_list)
        with open(user_1hop_file, 'w') as f:
            f.writelines(user_1hop_lines)
        with open(item_1hop_file, 'w') as f:
            f.writelines(item_1hop_lines)
        print('gen 1 hop files completed')


def gen_2hop(user_in_file = '../../data/douban/score-data/user-side.txt', item_in_file = '../../data/douban/score-data/item-side.txt',  
             item_1hop_file = '../../data/douban/score-data/item-1hop.txt', user_1hop_file = '../../data/douban/score-data/user-1hop.txt',
             item_2hop_file = '../../data/douban/score-data/item-2hop.txt', user_2hop_file = '../../data/douban/score-data/user-2hop.txt'):
    # load user-seq and item-seq
    user_seq_dict = {}
    item_seq_dict = {}

    with open(user_in_file) as f:
        lines = f.readlines()
        for line in lines:
            uid, useq = line[:-1].split('\t')[0], line[:-1].split('\t')[1:]
            user_seq_dict[uid] = useq
    
    with open(item_in_file) as f:
        lines = f.readlines()
        for line in lines:
            iid, iseq = line[:-1].split('\t')[0], line[:-1].split('\t')[1:]
            item_seq_dict[iid] = iseq
    
    # gen user side 2 hop
    user_2hop_lines = []
    with open(user_1hop_file) as f:
        lines = f.readlines()
        for line in lines:
            uid = line[:-1].split('\t')[0]
            list_1hop = line[:-1].split('\t')[1:]
            user_2hop_list = ['0,0,0' for j in range(len(list_1hop))]
            for i in range(len(list_1hop)):
                if list_1hop[i] == '0,0,0':
                    continue
                user_2hop = []
                time = list_1hop[i].split(' ')[0].split(',')[-1]
                for tup_str in list_1hop[i].split(' '):
                    iid = tup_str.split(',')[0]
                    iseq_list = item_seq_dict[iid]
                    for itup in iseq_list:
                        if itup.split(' ')[0].split(',')[-1] == time:
                            left = []
                            for t in itup.split(' '):
                                if uid not in t:
                                    left.append(t)
                            if left != []:
                                res = ' '.join(left)
                                user_2hop += [res]
                                break
                if user_2hop != []:
                    user_2hop = ' '.join(user_2hop)
                    if len(user_2hop.split(' ')) > K:
                        user_2hop_list[i] = ' '.join(user_2hop.split(' ')[:K])
                    else:
                        user_2hop_list[i] = user_2hop
    
            user_2hop_line = '\t'.join(user_2hop_list) + '\n'
            user_2hop_lines.append(user_2hop_line)
    
    with open(user_2hop_file, 'w') as f:
        f.writelines(user_2hop_lines)
    print('user 2 hop file completed')


    # gen item side 2 hop
    item_2hop_lines = []
    with open(item_1hop_file) as f:
        lines = f.readlines()
        for line in lines:
            iid = line[:-1].split('\t')[0]
            list_1hop = line[:-1].split('\t')[1:]
            item_2hop_list = ['0,0,0' for j in range(len(list_1hop))]
            for i in range(len(list_1hop)):
                if list_1hop[i] == '0,0,0':
                    continue
                item_2hop = []
                time = list_1hop[i].split(' ')[0].split(',')[-1]
                for tup_str in list_1hop[i].split(' '):
                    uid = tup_str.split(',')[0]
                    useq_list = user_seq_dict[uid]
                    for utup in useq_list:
                        if utup.split(' ')[0].split(',')[-1] == time:
                            left = []
                            for t in utup.split(' '):
                                if iid not in t:
                                    left.append(t)
                            if left != []:
                                res = ' '.join(left)
                                item_2hop += [res]
                                break
                if item_2hop != []:
                    item_2hop = ' '.join(item_2hop)
                    if len(item_2hop) > K:
                        item_2hop_list[i] = ' '.join(item_2hop.split(' ')[:K])
                    else:
                        item_2hop_list[i] = item_2hop
    
            item_2hop_line = '\t'.join(item_2hop_list) + '\n'
            item_2hop_lines.append(item_2hop_line)
    
    with open(item_2hop_file, 'w') as f:
        f.writelines(item_2hop_lines)
    print('item 2 hop file completed')

if __name__ == '__main__':
    # remap()
    # split_pos_neg()
    gen_user_item_seq()
    gen_1hop()
    gen_2hop()
