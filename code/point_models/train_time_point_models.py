import os
import tensorflow as tf
import sys
from data_loader import *
from point_model import *
from sklearn.metrics import *
import random
import time
import numpy as np
import pickle as pkl
import math

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2
EVAL_BATCH_SIZE = 1000
TRAIN_NEG_SAMPLE_NUM = 1
TEST_NEG_SAMPLE_NUM = 99


# for CCMR
FEAT_SIZE_CCMR = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1)
DATA_DIR_CCMR = '../../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 300

# for Taobao
FEAT_SIZE_Taobao = 1 + 984105 + 4067842 + 9405
DATA_DIR_Taobao = '../../../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 300

# for Tmall
FEAT_SIZE_Tmall = 1529672
DATA_DIR_Tmall = '../../../score-data/Tmall/feateng/'
MAX_LEN_Tmall = 300

def restore(data_set, target_file_test, user_seq_file_test,
        model_type, train_batch_size, feature_size, eb_dim, hidden_size, max_time_len, 
        lr, reg_lambda):
    print('restore begin')
    if model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_time_len)
    elif model_type == 'Caser': 
        model = Caser(feature_size, eb_dim, hidden_size, max_time_len)
    elif model_type == 'ARNN': 
        model = ARNN(feature_size, eb_dim, hidden_size, max_time_len)
    elif model_type == 'SVD++': 
        model = SVDpp(feature_size, eb_dim, hidden_size, max_time_len)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        logloss, auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss = eval(model, sess, target_file_test, max_time_len, reg_lambda, user_seq_file_test, 'restore')
        p = 1. / (1 + TEST_NEG_SAMPLE_NUM)
        rig = 1 -(logloss / -(p * math.log(p) + (1 - p) * math.log(1 - p)))
        print('RESTORE, LOSS TEST: %.4f  LOGLOSS TEST: %.4f  RIG TEST: %.4f  AUC TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f' % (loss, logloss, rig, auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr))

def get_ndcg(preds, target_iids):
    preds = np.array(preds).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1)[:,0].flatten().tolist()
    ndcg_val = []
    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        ndcg_val.append(getNDCG_at_K(ranklist, pos_iids[i], 5))
    return np.mean(ndcg_val)

def getNDCG_at_K(ranklist, target_item, k):
    for i in range(k):
        if ranklist[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def getHR_at_K(ranklist, target_item, k):
    if target_item in ranklist[:k]:
        return 1
    else:
        return 0

def getMRR(ranklist, target_item):
    for i in range(len(ranklist)):
        if ranklist[i] == target_item:
            return 1. / (i+1)
    return 0

def get_ranking_quality(preds, target_iids):
    preds = np.array(preds).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1)[:,0].flatten().tolist()
    ndcg_5_val = []
    ndcg_10_val = []
    hr_1_val = []
    hr_5_val = []
    hr_10_val = []
    mrr_val = []

    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        target_item = pos_iids[i]
        ndcg_5_val.append(getNDCG_at_K(ranklist, target_item, 5))
        ndcg_10_val.append(getNDCG_at_K(ranklist, target_item, 10))
        hr_1_val.append(getHR_at_K(ranklist, target_item, 1))
        hr_5_val.append(getHR_at_K(ranklist, target_item, 5))
        hr_10_val.append(getHR_at_K(ranklist, target_item, 10))
        mrr_val.append(getMRR(ranklist, target_item))
    return np.mean(ndcg_5_val), np.mean(ndcg_10_val), np.mean(hr_1_val), np.mean(hr_5_val), np.mean(hr_10_val), np.mean(mrr_val)


def eval(model, sess, target_file, max_time_len, reg_lambda, user_seq_file, mode = 'train'):
    preds = []
    labels = []
    target_iids = []
    losses = []

    data_loader = DataLoaderUserSeq(EVAL_BATCH_SIZE, max_time_len, target_file, user_seq_file, TEST_NEG_SAMPLE_NUM)
    t = time.time()
    for batch_data in data_loader:
        pred, label, loss = model.eval(sess, batch_data, reg_lambda)
        preds += pred
        labels += label
        losses.append(loss)
        target_iids += np.array(batch_data[3]).tolist()
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    if mode == 'train':
        ndcg = get_ndcg(preds, target_iids)
        print("EVAL TIME: %.4fs" % (time.time() - t))
        return logloss, auc, ndcg, loss
    elif mode == 'restore':
        ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr = get_ranking_quality(preds, target_iids)
        print("EVAL TIME: %.4fs" % (time.time() - t))
        return logloss, auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss

def train(data_set, target_file_train, target_file_test, user_seq_file_train, user_seq_file_test,
        model_type, train_batch_size, feature_size, 
        eb_dim, hidden_size, max_time_len, lr, reg_lambda, dataset_size):
    if model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_time_len)
    elif model_type == 'Caser': 
        model = Caser(feature_size, eb_dim, hidden_size, max_time_len)
    elif model_type == 'ARNN': 
        model = ARNN(feature_size, eb_dim, hidden_size, max_time_len)
    elif model_type == 'SVD++': 
        model = SVDpp(feature_size, eb_dim, hidden_size, max_time_len)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    
    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []
        
        train_losses = []
        test_loglosses = []
        test_aucs = []
        test_ndcgs = []
        test_losses = []

        # before training process
        step = 0
        test_logloss, test_auc, test_ndcg, test_loss = eval(model, sess, target_file_test, max_time_len, reg_lambda, user_seq_file_test)
        test_loglosses.append(test_logloss)
        test_aucs.append(test_auc)
        test_ndcgs.append(test_ndcg)
        test_losses.append(test_loss)

        print("STEP %d LOSS TRAIN: NaN  LOSS TEST: %.4f  LOGLOSS TEST: %.4f  AUC TEST: %.4f  NDCG@5 TEST: %.4f" % (step, test_loss, test_logloss, test_auc, test_ndcg))
        early_stop = False
        eval_iter_num = (dataset_size // 5) // (train_batch_size / (1 + TRAIN_NEG_SAMPLE_NUM))
        # begin training process
        for epoch in range(5):
            if early_stop:
                break
            data_loader = DataLoaderUserSeq(train_batch_size, max_time_len, target_file_train, user_seq_file_train, TRAIN_NEG_SAMPLE_NUM)
            for batch_data in data_loader:
                if early_stop:
                    break
                loss = model.train(sess, batch_data, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []
                    test_logloss, test_auc, test_ndcg, test_loss = eval(model, sess, target_file_test, max_time_len, reg_lambda, user_seq_file_test)

                    test_loglosses.append(test_logloss)
                    test_aucs.append(test_auc)
                    test_ndcgs.append(test_ndcg)
                    test_losses.append(test_loss)

                    print("STEP %d  LOSS TRAIN: %.4f  LOSS TEST: %.4f  LOGLOSS TEST: %.4f  AUC TEST: %.4f  NDCG@5 TEST: %.4f" % (step, train_loss, test_loss, test_logloss, test_auc, test_ndcg))
                    if test_losses[-1] < min(test_losses[:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/'.format(data_set, model_name)):
                            os.makedirs('save_model_{}/{}/'.format(data_set, model_name))
                        save_path = 'save_model_{}/{}/ckpt'.format(data_set, model_name)
                        model.save(sess, save_path)

                    if len(test_losses) > 2 and epoch > 0:
                        if (test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3]):
                            early_stop = True

        # generate log
        if not os.path.exists('logs_{}/'.format(data_set)):
            os.makedirs('logs_{}/'.format(data_set))
        model_name = '{}_{}_{}'.format(model_type, lr, reg_lambda)

        with open('logs_{}/{}.pkl'.format(data_set, model_name), 'wb') as f:
            dump_tuple = (train_losses, test_losses, test_loglosses, test_aucs, test_ndcgs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, model_name), 'w') as f:
            index = np.argmin(test_losses)
            f.write('Result Test AUC: {}\n'.format(test_aucs[index]))
            f.write('Result Test Logloss: {}\n'.format(test_loglosses[index]))
            f.write('Result Test NDCG@5: {}\n'.format(test_ndcgs[index]))
        return 
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    model_type = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    data_set = sys.argv[3]

    if data_set == 'ccmr':
        target_file_train = DATA_DIR_CCMR + 'target_39_hot_sample.txt'
        target_file_test = DATA_DIR_CCMR + 'target_40_hot_sample.txt'
        user_seq_file_train = DATA_DIR_CCMR + 'train_user_hist_seq_39_sample.txt'
        user_seq_file_test = DATA_DIR_CCMR + 'test_user_hist_seq_40_sample.txt'
        # model parameter
        feature_size = FEAT_SIZE_CCMR
        max_time_len = MAX_LEN_CCMR
        dataset_size = 262090
    elif data_set == 'taobao':
        target_file_train = DATA_DIR_Taobao + 'target_train4point_model_sample.txt'
        target_file_test = DATA_DIR_Taobao + 'target_17_hot_sample.txt'
        user_seq_file_train = DATA_DIR_Taobao + 'train_user_hist_seq_sample.txt'
        user_seq_file_test = DATA_DIR_Taobao + 'test_user_hist_seq_sample.txt'
        # model parameter
        feature_size = FEAT_SIZE_Taobao
        max_time_len = MAX_LEN_Taobao
        dataset_size = 262988
    elif data_set == 'tmall':
        target_file_train = DATA_DIR_Tmall + 'target_10_hot_sample.txt'
        target_file_test = DATA_DIR_Tmall + 'target_11_hot_sample.txt'
        user_seq_file_train = DATA_DIR_Tmall + 'train_user_hist_seq_sample.txt'
        user_seq_file_test = DATA_DIR_Tmall + 'test_user_hist_seq_sample.txt'
        # model parameter
        feature_size = FEAT_SIZE_Tmall
        max_time_len = MAX_LEN_Tmall
        dataset_size = 228213
    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    reg_lambda = 1e-3
    hyper_paras = [(100, 1e-4), (200, 5e-4)]

    for hyper in hyper_paras:
        train_batch_size, lr = hyper
        train(data_set, target_file_train, target_file_test, user_seq_file_train, user_seq_file_test,
                model_type, train_batch_size, feature_size, 
                EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, lr, reg_lambda, dataset_size)
        
        restore(data_set, target_file_test, user_seq_file_test,
            model_type, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, 
            lr, reg_lambda)