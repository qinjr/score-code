import pickle as pkl
import time
import numpy as np


DATA_DIR_CCMR = '../../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 300

DATA_DIR_Taobao = '../../../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 300

class DataLoaderUserSeq(object):
    def __init__(self, batch_size, max_len, target_file, user_seq_file, neg_sample_num):
        self.batch_size = batch_size
        self.max_len = max_len
        self.neg_sample_num = neg_sample_num
        if self.batch_size % (1 + self.neg_sample_num) != 0:
            print('batch size should be time of {}'.format(1 + self.neg_sample_num))
            exit(1)
        self.batch_size2line_num = int(self.batch_size / (1 + self.neg_sample_num))

        self.target_f = open(target_file)
        self.user_seq_f = open(user_seq_file)

    def __iter__(self):
        return self
    
    def __next__(self):
        target_user_batch = []
        target_item_batch = []
        label_batch = []
        user_seq_batch = []
        user_seq_len_batch = []

        for i in range(self.batch_size2line_num):
            target_line = self.target_f.readline()
            if target_line == '':
                raise StopIteration
            user_seq_line = self.user_seq_f.readline()
            target_line_split_list = target_line[:-1].split(',')
            uid, iids = target_line_split_list[0], target_line_split_list[1:(2 + self.neg_sample_num)]
            
            user_seq_list = [iid for iid in user_seq_line[:-1].split(',')]
            user_seq_one = []
            
            for iid in user_seq_list:
                user_seq_one.append(int(iid))
            for p in range(self.max_len - len(user_seq_one)):
                user_seq_one.append(0)

            for j in range(len(iids)):
                if j == 0:
                    label_batch.append(1)
                else:
                    label_batch.append(0)
                target_user_batch.append(int(uid))
                target_item_batch.append(int(iids[j]))
                user_seq_len_batch.append(len(user_seq_list))
                user_seq_batch.append(user_seq_one)
                
        return user_seq_batch, user_seq_len_batch, target_user_batch, target_item_batch, label_batch

class DataLoaderDualSeq(object):
    def __init__(self, batch_size, max_len, target_file, user_seq_file, item_seq_file, neg_sample_num):
        self.batch_size = batch_size
        self.max_len = max_len
        self.neg_sample_num = neg_sample_num
        if self.batch_size % (1 + self.neg_sample_num) != 0:
            print('batch size should be time of {}'.format(1 + self.neg_sample_num))
            exit(1)
        self.batch_size2line_num = int(self.batch_size / (1 + self.neg_sample_num))

        self.target_f = open(target_file)
        self.user_seq_f = open(user_seq_file)
        self.item_seq_f = open(item_seq_file)

    def __iter__(self):
        return self
    
    def __next__(self):
        target_user_batch = []
        target_item_batch = []
        label_batch = []
        user_seq_batch = []
        user_seq_len_batch = []
        item_seq_batch = []
        item_seq_len_batch = []

        for i in range(self.batch_size2line_num):
            target_line = self.target_f.readline()
            if target_line == '':
                raise StopIteration
            user_seq_line = self.user_seq_f.readline()
            item_seq_line = self.item_seq_f.readline()

            target_line_split_list = target_line[:-1].split(',')
            uid, iids = target_line_split_list[0], target_line_split_list[1:(2 + self.neg_sample_num)]
            
            user_seq_list = [iid for iid in user_seq_line[:-1].split(',')]
            user_seq_one = []
            item_seq_list = [uid for uid in item_seq_line[:-1].split(',')]
            item_seq_one = []

            for iid in user_seq_list:
                user_seq_one.append(int(iid))
            for uid in item_seq_list:
                item_seq_one.append(int(uid))

            for p in range(self.max_len - len(user_seq_one)):
                user_seq_one.append(0)
            for p in range(self.max_len - len(item_seq_one)):
                item_seq_one.append(0)

            for j in range(len(iids)):
                if j == 0:
                    label_batch.append(1)
                else:
                    label_batch.append(0)
                target_user_batch.append(int(uid))
                target_item_batch.append(int(iids[j]))
                user_seq_len_batch.append(len(user_seq_list))
                user_seq_batch.append(user_seq_one)
                item_seq_len_batch.append(len(item_seq_list))
                item_seq_batch.append(item_seq_one)
                
        return user_seq_batch, user_seq_len_batch, item_seq_batch, item_seq_len_batch, target_user_batch, target_item_batch, label_batch

if __name__ == "__main__":
    data_loader = DataLoaderUserSeq(100, 300, DATA_DIR_CCMR + 'target_39_hot_sample.txt',
                                    MAX_LEN_CCMR + 'train_user_hist_seq_39_sample.txt', 1)
    
    t = time.time()
    for batch_data in data_loader:
        print(np.array(batch_data[0]).shape)
        print(np.array(batch_data[1]).shape)
        print(np.array(batch_data[2]).shape)
        print(np.array(batch_data[3]).shape)
        print(np.array(batch_data[4]).shape)
        
        print('time of batch: {}'.format(time.time()-t))
        t = time.time()
        