import random
import sys

# CCMR dataset parameters
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'

# Taobao dataset parameters
DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'

# Tmall dataset parameters
DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'

def sample_files(target_file, user_seq_file, sample_target_file, sample_user_seq_file,
                sample_factor):
    target_lines = open(target_file).readlines()
    user_seq_lines = open(user_seq_file).readlines()

    sample_target_lines = []
    sample_user_seq_lines = []
    
    length = len(target_lines)
    for i in range(length):
        rand_int = random.randint(1, sample_factor)
        if rand_int == 1:
            sample_target_lines.append(target_lines[i])
            sample_user_seq_lines.append(user_seq_lines[i])
    with open(sample_target_file, 'w') as f:
        f.writelines(sample_target_lines)
    with open(sample_user_seq_file, 'w') as f:
        f.writelines(sample_user_seq_lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # CCMR
        sample_files(DATA_DIR_CCMR + 'target_39_hot.txt', DATA_DIR_CCMR + 'train_user_hist_seq_39.txt', DATA_DIR_CCMR + 'target_39_hot_sample.txt', DATA_DIR_CCMR + 'train_user_hist_seq_39_sample.txt', 2)
        sample_files(DATA_DIR_CCMR + 'target_40_hot.txt', DATA_DIR_CCMR + 'test_user_hist_seq_40.txt', DATA_DIR_CCMR + 'target_40_hot_sample.txt', DATA_DIR_CCMR + 'test_user_hist_seq_40_sample.txt', 10)
    elif dataset == 'taobao':
        # CCMR
        sample_files(DATA_DIR_Taobao + 'target_7_hot.txt', DATA_DIR_Taobao + 'train_user_hist_seq_7.txt', DATA_DIR_Taobao + 'target_7_hot_sample.txt', DATA_DIR_Taobao + 'train_user_hist_seq_7_sample.txt', 2)
        sample_files(DATA_DIR_Taobao + 'target_8_hot.txt', DATA_DIR_Taobao + 'test_user_hist_seq_8.txt', DATA_DIR_Taobao + 'target_8_hot_sample.txt', DATA_DIR_Taobao + 'test_user_hist_seq_8_sample.txt', 6)
    elif dataset == 'tmall':
        # CCMR
        sample_files(DATA_DIR_Tmall + 'target_10_hot.txt', DATA_DIR_Tmall + 'train_user_hist_seq_10.txt', DATA_DIR_Tmall + 'target_10_hot_sample.txt', DATA_DIR_Tmall + 'train_user_hist_seq_10_sample.txt', 1)
        sample_files(DATA_DIR_Tmall + 'target_11_hot.txt', DATA_DIR_Tmall + 'test_user_hist_seq_11.txt', DATA_DIR_Tmall + 'target_11_hot_sample.txt', DATA_DIR_Tmall + 'test_user_hist_seq_11_sample.txt', 6)
    else:
        print('WRONG DATASET: {}'.format(dataset))

