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
            time_str = line[:-1].split(',')[3]
            time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
            times.append(times)
    start_time = min(times)
    t_idx = [(t - start_time) / (24 * 3600 * TIME_DELTA) for t in times]
    print('max time idx: {}'.format(max(t_idx)))

    plt.hist(t_idx, bins=range(max(t_idx)+1))
    plt.savefig(plt_file)

def movie_feat_info(in_file):
    field_dict = {
        'director': [],
        'actor': [],
        'genre': [],
        'nation': []
    }
    director_width, actor_width, genre_width, nation_width = 0, 0, 0, 0
    with open(in_file, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            _, directors, actors, genres, nations, __ = line.split(',')
            director_list = directors.split(';')
            field_dict['director'] += director_list
            if len(director_list) > director_width:
                director_width = len(director_list)
            
            actor_list = actors.split(';')
            field_dict['actor'] += actor_list
            if len(actor_list) > actor_width:
                actor_width = len(actor_list)
            
            genre_list = genres.split(';')
            field_dict['genre'] += genre_list
            if len(genre_list) > genre_width:
                genre_width = len(genre_list)
            
            nation_list = nations.split(';')
            field_dict['nation'] += nation_list
            if len(nation_list) > nation_width:
                nation_width = len(nation_list)
    
    for key in field_dict:
        field_dict[key] = set(field_dict[key])
        print(key, len(field_dict[key]))
        print(key, max(field_dict[key]))
        print(key, min(field_dict[key]))
    print(director_width, actor_width, genre_width, nation_width)

if __name__ == "__main__":
    # pos_neg_split(RAW_DIR + 'rating_logs.csv', FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'rating_neg.csv')
    # time_distri(FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'time_distri.png')
    movie_feat_info(RAW_DIR + 'movie_info.csv')