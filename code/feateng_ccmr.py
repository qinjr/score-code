import pickle as pkl
import datetime
import time
import matplotlib.pyplot as plt

RAW_DIR = '../../score-data/CCMR/raw_data/'
FEATENG_DIR = '../../score-data/CCMR/feateng/'

TIME_DELTA = 180

def pos_neg_split(in_file, pos_file, neg_file):
    pos_lines = []
    neg_lines = []
    with open(in_file, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            else:
                if line.split(',')[2] == '5' or line.split(',')[2] == '4':
                    pos_lines.append(line)
                else:
                    neg_lines.append(line)
    
    print('pos sample: {}'.format(len(pos_lines)))
    print('neg sample: {}'.format(len(neg_lines)))
    with open(pos_file, 'w') as f:
        f.writelines(pos_lines)
    with open(neg_file, 'w') as f:
        f.writelines(neg_lines)

def time_distri(in_file, plt_file):
    times = []
    with open(in_file, 'r') as f:
        for line in f:
            time_str = line.split(',')[3]
            time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
            times.append(times)
    start_time = min(times)
    t_idx = [(t - start_time) / (24 * 3600 * TIME_DELTA) for t in times]
    print('max time idx: {}'.format(max(t_idx)))

    plt.hist(t_idx, bins=range(max(t_idx)+1))
    plt.savefig(plt_file)


if __name__ == "__main__":
    # pos_neg_split(RAW_DIR + 'rating_logs.csv', FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'rating_neg.csv')
    time_distri(FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'time_distri.png')
