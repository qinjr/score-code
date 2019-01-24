import numpy as np
import random

class DataLoader_SCORE:
    def __init__(self, batch_size, item_1hop_file, user_1hop_file, 
                 item_2hop_file, user_2hop_file, target_file, feature_size, max_len, k):
        self.batch_size = batch_size
        with open(target_file, 'r') as f:
            self.target_list = f.readlines()
            f.close()

        self.num_of_step = len(self.target_list) / self.batch_size

        self.user_1hop_f = open(user_1hop_file)
        self.user_2hop_f = open(user_2hop_file)

        self.item_1hop_f = open(item_1hop_file)
        self.item_2hop_f = open(item_2hop_file)
        self.target_f = open(target_file)

        self.null = feature_size - 1
        self.numb = feature_size - 2
        self.max_len = max_len

        self.s = 0

        # each time slice, we choose 5 instances at most, if less than K, use null representation
        self.K = k

    def __iter__(self):
        return self

    def next(self):
        if self.s == self.num_of_step:
            raise StopIteration

        num_of_line = self.batch_size
        # if len(self.target_list) - self.i * self.batch_size < self.batch_size:
        #     num_of_line = len(self.target_list) - self.i * self.batch_size

        label, target_user, target_item, user_1hop, user_2hop, item_1hop, item_2hop, user_1hop_len, user_2hop_len, item_1hop_len, item_2hop_len, \
        user_1hop_border, user_2hop_border, item_1hop_border, item_2hop_border = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(num_of_line):
            target_line = self.target_f.readline()[:-1]
            l, tgt_u, tgt_i = int(target_line.split(',')[-1]), int(target_line.split(',')[0]), [int(target_line.split(',')[1]), int(target_line.split(',')[2])]
            label.append(l)
            target_user.append([tgt_u])
            target_item.append(tgt_i)

            user_1hop_sample, user_2hop_sample, item_1hop_sample, item_2hop_sample = [], [], [], []
            user_1hop_border_sample, user_2hop_border_sample, item_1hop_border_sample, item_2hop_border_sample = [], [], [], []

            # user_1hop
            user_1hop_line_list = self.user_1hop_f.readline()[:-1].split('\t')[1:]
            for t in user_1hop_line_list:
                if t == '0,0,0':
                    user_1hop_sample.append([[self.numb, self.numb] for k in range(self.K)])
                    user_1hop_border_sample.append([self.K] * self.K)
                    continue

                user_1hop_t = []
                t_list = t.split(' ')
                t_list_len = len(t_list)

                for j in range(t_list_len):
                    user_1hop_t.append([int(t_list[j].split(',')[0]), int(t_list[j].split(',')[1])])
                for j in range(self.K - t_list_len):
                    user_1hop_t.append([self.null, self.null])
                    # rand = random.randint(0, t_list_len - 1)
                    # user_1hop_t.append([int(t_list[rand].split(',')[0]), int(t_list[rand].split(',')[1])])
                user_1hop_border_sample.append([t_list_len] * self.K)
                # user_1hop_border_sample.append([self.K] * self.K)
                user_1hop_sample.append(user_1hop_t)
            
            user_1hop_len.append(len(user_1hop_line_list))
            # padding
            user_1hop_sample = user_1hop_sample + [user_1hop_sample[-1] for k in range(self.max_len - len(user_1hop_sample))]
            user_1hop.append(user_1hop_sample)
            user_1hop_border_sample = user_1hop_border_sample + [user_1hop_border_sample[-1] for k in range(self.max_len - len(user_1hop_border_sample))]
            user_1hop_border.append(user_1hop_border_sample)

            # user_2hop
            user_2hop_line_list = self.user_2hop_f.readline()[:-1].split('\t')
            for t in user_2hop_line_list:
                if t == '0,0,0':
                    user_2hop_sample.append([[self.numb] for k in range(self.K)])
                    user_2hop_border_sample.append([self.K] * self.K)
                    continue
                
                user_2hop_t = []
                t_list = t.split(' ')
                t_list_len = len(t_list)
            
                for j in range(t_list_len):
                    user_2hop_t.append([int(t_list[j].split(',')[0])])
                for j in range(self.K - t_list_len):
                    user_2hop_t.append([self.null])
                    # rand = random.randint(0, t_list_len - 1)
                    # user_2hop_t.append([int(t_list[rand].split(',')[0])])
                user_2hop_border_sample.append([t_list_len] * self.K)
                # user_2hop_border_sample.append([self.K] * self.K)
                user_2hop_sample.append(user_2hop_t)
            
            user_2hop_len.append(len(user_2hop_line_list))
            if user_2hop_sample == []:
                print(user_2hop_line)
            # padding
            user_2hop_sample = user_2hop_sample + [user_2hop_sample[-1] for k in range(self.max_len - len(user_2hop_sample))]
            user_2hop.append(user_2hop_sample)
            user_2hop_border_sample = user_2hop_border_sample + [user_2hop_border_sample[-1] for k in range(self.max_len - len(user_2hop_border_sample))]
            user_2hop_border.append(user_2hop_border_sample)

            # item_1hop
            item_1hop_line_list = self.item_1hop_f.readline()[:-1].split('\t')[1:]
            for t in item_1hop_line_list:
                if t == '0,0,0':
                    item_1hop_sample.append([[self.numb] for k in range(self.K)])
                    item_1hop_border_sample.append([self.K] * self.K)
                    continue

                item_1hop_t = []
                t_list = t.split(' ')
                t_list_len = len(t_list)
                for j in range(t_list_len):
                    item_1hop_t.append([int(t_list[j].split(',')[0])])
                for j in range(self.K - t_list_len):
                    item_1hop_t.append([self.null])
                    # rand = random.randint(0, t_list_len - 1)
                    # item_1hop_t.append([int(t_list[rand].split(',')[0])])
                item_1hop_border_sample.append([t_list_len] * self.K)
                # item_1hop_border_sample.append([self.K] * self.K)
                item_1hop_sample.append(item_1hop_t)

            item_1hop_len.append(len(item_1hop_line_list))
            # padding
            item_1hop_sample = item_1hop_sample + [item_1hop_sample[-1] for k in range(self.max_len - len(item_1hop_sample))]
            item_1hop.append(item_1hop_sample)
            item_1hop_border_sample = item_1hop_border_sample + [item_1hop_border_sample[-1] for k in range(self.max_len - len(item_1hop_border_sample))]
            item_1hop_border.append(item_1hop_border_sample)

            # item_2hop
            item_2hop_line_list = self.item_2hop_f.readline()[:-1].split('\t')
            for t in item_2hop_line_list:
                if t == '0,0,0':
                    item_2hop_sample.append([[self.numb, self.numb] for k in range(self.K)])
                    item_2hop_border_sample.append([self.K] * self.K)
                    continue
                
                item_2hop_t = []
                t_list = t.split(' ')
                t_list_len = len(t_list)
                for j in range(t_list_len):
                    item_2hop_t.append([int(t_list[j].split(',')[0]), int(t_list[j].split(',')[1])])
                for j in range(self.K - t_list_len):
                    item_2hop_t.append([self.null, self.null])
                    # rand = random.randint(0, t_list_len - 1)
                    # item_2hop_t.append([int(t_list[rand].split(',')[0]), int(t_list[rand].split(',')[1])])
                item_2hop_border_sample.append([t_list_len] * self.K)
                # item_2hop_border_sample.append([self.K] * self.K)
                item_2hop_sample.append(item_2hop_t)
            
            item_2hop_len.append(len(item_2hop_line_list))
            # padding
            item_2hop_sample = item_2hop_sample + [item_2hop_sample[-1] for k in range(self.max_len - len(item_2hop_sample))]
            item_2hop.append(item_2hop_sample)
            item_2hop_border_sample = item_2hop_border_sample + [item_2hop_border_sample[-1] for k in range(self.max_len - len(item_2hop_border_sample))]
            item_2hop_border.append(item_2hop_border_sample)

        self.s += 1
        return [label, target_user, target_item, user_1hop, user_2hop, item_1hop, item_2hop, user_1hop_len, user_2hop_len, item_1hop_len, item_2hop_len, \
               user_1hop_border, user_2hop_border, item_1hop_border, item_2hop_border]

class DataLoader_RNN:
    def __init__(self, batch_size, user_1hop_file, item_1hop_file,
                 target_file, feature_size, max_len, k):
        self.batch_size = batch_size
        with open(target_file, 'r') as f:
            self.target_list = f.readlines()
            f.close()

        self.num_of_step = len(self.target_list) / self.batch_size

        self.user_1hop_f = open(user_1hop_file)
        self.item_1hop_f = open(item_1hop_file)
        self.target_f = open(target_file)
        self.max_len = max_len
        self.s = 0

    def __iter__(self):
        return self

    def next(self):
        if self.s == self.num_of_step:
            raise StopIteration

        num_of_line = self.batch_size

        label, target_user, target_item, user_1hop, user_1hop_len, item_1hop, item_1hop_len = [], [], [], [], [], [], []

        for i in range(num_of_line):
            target_line = self.target_f.readline()[:-1]
            l, tgt_u, tgt_i = int(target_line.split(',')[-1]), int(target_line.split(',')[0]), [int(target_line.split(',')[1]), int(target_line.split(',')[2])]
            label.append(l)
            target_user.append([tgt_u])
            target_item.append(tgt_i)

            user_1hop_sample = []

            user_1hop_line_list = self.user_1hop_f.readline()[:-1].split()[1:]
            for t in user_1hop_line_list:
                if t == '0,0,0':
                    continue
                else:
                    user_1hop_sample.append([int(t.split(',')[0]), int(t.split(',')[1])])
            if len(user_1hop_sample) >= self.max_len:
                user_1hop_len.append(self.max_len)
                user_1hop_sample = user_1hop_sample[-self.max_len:]
            else:
                user_1hop_len.append(len(user_1hop_sample))
                user_1hop_sample = user_1hop_sample + [user_1hop_sample[-1] for p in range(self.max_len - len(user_1hop_sample))]
            user_1hop.append(user_1hop_sample)

            item_1hop_sample = []

            item_1hop_line_list = self.item_1hop_f.readline()[:-1].split()[1:]
            for t in item_1hop_line_list:
                if t == '0,0,0':
                    continue
                else:
                    item_1hop_sample.append([int(t.split(',')[0])])
            if len(item_1hop_sample) >= self.max_len:
                item_1hop_len.append(self.max_len)
                item_1hop_sample = item_1hop_sample[-self.max_len:]
            else:
                item_1hop_len.append(len(item_1hop_sample))
                item_1hop_sample = item_1hop_sample + [item_1hop_sample[-1] for p in range(self.max_len - len(item_1hop_sample))]
            item_1hop.append(item_1hop_sample)

        self.s += 1
        return [label, target_user, target_item, user_1hop, user_1hop_len, item_1hop, item_1hop_len]



if __name__ == "__main__":
    dataloader = DataLoader_Neg(128)
    i = 0
    # for batch in dataloader:
    batch = dataloader.next()
    print(np.array(batch[0]).shape)

