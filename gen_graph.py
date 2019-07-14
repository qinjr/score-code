import networkx as nx
import os
import sys
import pickle as pkl
import time

TIME_SLICE_NUM_CCMR = 41
TIME_SLICE_NUM_Taobao = 9
TIME_SLICE_NUM_Tmall = 14

RATING_FILE_CCMR = '../score-data/CCMR/feateng/remap_rating_pos_idx.csv'
RATING_FILE_Taobao = '../score-data/Taobao/feateng/remaped_user_behavior.txt'
RATING_FILE_Tmall = '../score-data/Tmall/feateng/remaped_user_behavior.txt'

def gen_adj_dicts(in_file, total_slice, dataset_name):
    # all adj dicts for each temporal graph 
    adj_dicts = []
    for i in range(total_slice):
        adj_dicts.append({})
    # make adj dicts dir
    dir_name = '../score-data/{}/adj_dicts'.format(dataset_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    t = time.time()
    # gen adj dicts
    with open(in_file) as f:
        for line in f:
            uid, iid, _, idx = line[:-1].split(',')
            adj_dict = adj_dicts[int(idx)]
            if uid not in adj_dict:
                adj_dict[uid] = [iid]
            else:
                adj_dict[uid].append(iid)
    print('adj dicts completed, time cost: %.1f minutes' % ((time.time() - t) / 60))

    # dump dicts
    t = time.time()
    dump_file_name = '{}/adj_dicts.pkl'.format(dir_name)
    with open(dump_file_name, 'wb') as f:
        pkl.dump(adj_dicts, f)
    print('all adj dict files for %s dataset have been generated, dump time cost: %.1f minutes' % (dataset_name, (time.time() - t)/60))

def gen_adj_lists(total_slice, dataset_name):
    # make adj lines dir
    dir_name = '../score-data/{}/adj_lines/'.format(dataset_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # load adj dicts
    dicts_file = '../score-data/{}/adj_dicts/adj_dicts.pkl'.format(dataset_name)
    with open(dicts_file, 'rb') as f:
        adj_dicts = pkl.load(f)
    
    # gen lines and write adj lines files
    for i in range(total_slice):
        t = time.time()
        adj_lines = []
        adj_dict = adj_dicts[i]
        for key in adj_dict.keys():
            adj_line = ' '.join([key] + adj_dict[key]) + '\n'
            adj_lines.append(adj_line)
        with open(dir_name + 'adj_line_{}.txt'.format(i), 'w') as f:
            f.writelines(adj_lines)
        print('temporal graph %d completed, time cost: %.2f seconds' % (i, time.time() - t))
    print('adj lines files all generated')

def gen_graph(total_slice, dataset_name):
    # make graph dir
    dir_name = '../score-data/{}/graphs/'.format(dataset_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    for i in range(total_slice):
        t = time.time()
        adj_file_name = '../score-data/{}/adj_lines/adj_line_{}.txt'.format(dataset_name, i)
        G = nx.read_adjlist(adj_file_name)
        print('num of nodes: {}  num of edges: {}'.format(G.number_of_nodes(), G.number_of_edges()))
        print('temporal graph %d completed, time cost: %.2f seconds' % (i, time.time() - t))
        
        # dump the networkx Graph obj for further use
        graph_name = dir_name + 'graph_{}.pkl'.format(i)
        with open(graph_name, 'wb') as f:
            pkl.dump(G, f)


if __name__ == '__main__':
    # gen_adj_dicts(RATING_FILE_CCMR, TIME_SLICE_NUM_CCMR, 'CCMR')
    # gen_adj_lists(TIME_SLICE_NUM_CCMR, 'CCMR')
    gen_graph(TIME_SLICE_NUM_CCMR, 'CCMR')
