import cPickle as pkl
import datetime

RAW_DIR = '../../score-data/CCMR/raw_data/'
FEATENG_DIR = '../../score-data/CCMR/feateng/'

def pos_neg_split(in_file, pos_file, neg_file):
    newlines = []
    with open(in_file, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            else:
                if line.split(',')[2] == '5' or line.split(',')[2] == '4':
                    newlines.append(line)
                    