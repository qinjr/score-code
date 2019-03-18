import os
import tensorflow as tf
import sys
from data_loader import *
from model import *
from sklearn.metrics import *
import random
import time
import numpy as np
import cPickle as pkl
import math

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2

EVAL_BATCH_SIZE = 2000
NEG_NUM = 4


def restore(item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test,
        model_type, train_num, batch_size, feature_size, eb_dim, hidden_size, max_len_point, max_len_slice, k, user_fnum, item_fnum, lr, reg_lambda, data_set):
    print('restore begin')
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'SCORE_B':
        model = SCORE_B(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'SCORE_Att':
        model = SCORE_Att(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'RRN':
        model = RRN(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'RRN_WS':
        model = RRN_WS(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'RRN_2HOP':
        model = RRN_2HOP(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_len_point, user_fnum, item_fnum)
    elif model_type == 'GRU4Rec_DUAL':
        model = GRU4Rec_DUAL(feature_size, eb_dim, hidden_size, max_len_point, user_fnum, item_fnum)
    elif model_type == 'SVD++':
        model = SVDpp(feature_size, eb_dim, max_len_point, user_fnum, item_fnum)
    elif model_type == 'GCMC':
        model = GCMC(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    
    if model_type in ['SCORE', 'SCORE_B', 'SCORE_Att', 'RRN', 'RRN_2HOP', 'RRN_WS', 'GCMC']:
        max_len = max_len_slice
    else:
        max_len = max_len_point
    
    model_name = '{}_{}_{}_{}_{}'.format(model_type, train_num, train_batch_size, lr, reg_lambda)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        logloss, auc, ndcg = eval(model, model_type, sess, item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test, feature_size, max_len, k)
        print('RESTORE, LOSS TEST: %.4f  AUC TEST: %.4f  NDCG@5 TEST: %.4f' % (logloss, auc, ndcg))

def getNDCG(ranklist, target_item):
    for i in range(len(ranklist)):
        if ranklist[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def get_ndcg(preds, target_iids):
    preds = np.array(preds).reshape(-1, NEG_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, NEG_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, NEG_NUM + 1)[:,0].flatten().tolist()
    ndcg_val = []
    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        ndcg_val.append(getNDCG(ranklist, pos_iids[i]))
    return np.mean(ndcg_val)

def eval(model, model_type, sess, item_1hop_file, user_1hop_file, item_2hop_file, user_2hop_file, target_file, feature_size, max_len, k):
    preds = []
    labels = []
    target_iids = []
    if model_type == 'SCORE' or model_type == 'SCORE_B' or model_type == 'SCORE_Att' or model_type == 'RRN' or model_type == 'RRN_WS' or model_type == 'RRN_2HOP' or model_type == 'GCMC':
        dataloader = DataLoader_SCORE(EVAL_BATCH_SIZE, item_1hop_file, user_1hop_file, item_2hop_file, user_2hop_file, target_file, feature_size, max_len, k)
    elif model_type == 'GRU4Rec' or model_type == 'GRU4Rec_DUAL' or model_type == 'SVD++':
        dataloader = DataLoader_RNN(EVAL_BATCH_SIZE, user_1hop_file, item_1hop_file, target_file, feature_size, max_len, k)
    t = time.time()
    for batch_data in dataloader:
        pred, label = model.eval(sess, batch_data)
        preds += pred
        labels += label
        target_iids += np.array(batch_data[2])[:,0].tolist()
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    ndcg = get_ndcg(preds, target_iids)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return logloss, auc, ndcg

def train(item_1hop_file_train, user_1hop_file_train, item_2hop_file_train, user_2hop_file_train, target_file_train,
        item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test,
        model_type, train_num, batch_size, feature_size, eb_dim, hidden_size, max_len_point, max_len_slice, k, user_fnum, item_fnum, lr, reg_lambda, data_set, eval_iter_num):
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'SCORE_B':
        model = SCORE_B(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'SCORE_Att':
        model = SCORE_Att(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'RRN':
        model = RRN(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'RRN_WS':
        model = RRN_WS(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'RRN_2HOP':
        model = RRN_2HOP(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    elif model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_len_point, user_fnum, item_fnum)
    elif model_type == 'GRU4Rec_DUAL':
        model = GRU4Rec_DUAL(feature_size, eb_dim, hidden_size, max_len_point, user_fnum, item_fnum)
    elif model_type == 'SVD++':
        model = SVDpp(feature_size, eb_dim, max_len_point, user_fnum, item_fnum)
    elif model_type == 'GCMC':
        model = GCMC(feature_size, eb_dim, hidden_size, max_len_slice, k, user_fnum, item_fnum)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    
    if model_type in ['SCORE', 'SCORE_B', 'SCORE_Att', 'RRN', 'RRN_2HOP', 'RRN_WS', 'GCMC']:
        max_len = max_len_slice
    else:
        max_len = max_len_point
    
    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses = []
        test_losses = []
        train_aucs = []
        test_aucs = []

        # before training process
        step = 0
        train_loss, train_auc, _ = eval(model, model_type, sess, item_1hop_file_train, user_1hop_file_train, item_2hop_file_train, user_2hop_file_train, target_file_train, feature_size, max_len, k)
        test_loss, test_auc, _ = eval(model, model_type, sess, item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test, feature_size, max_len, k)

        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        test_losses.append(test_loss)
        test_aucs.append(test_auc)

        print("STEP %d  LOSS_TRAIN: %.4f  LOSS_TEST: %.4f  AUC_TRAIN: %.4f  AUC_TEST: %.4f" % (step, train_loss, test_loss, train_auc, test_auc))
        early_stop = False

        # begin training process
        for epoch in range(3):
            if early_stop:
                break
            if model_type == 'SCORE' or model_type == 'SCORE_B' or model_type == 'SCORE_Att' or model_type == 'RRN' or model_type == 'RRN_WS' or model_type == 'RRN_2HOP' or model_type == 'GCMC':
                dataloader = DataLoader_SCORE(batch_size, item_1hop_file_train, user_1hop_file_train, item_2hop_file_train, user_2hop_file_train, target_file_train, feature_size, max_len, k)
            elif model_type == 'GRU4Rec' or model_type == 'GRU4Rec_DUAL' or model_type == 'SVD++':
                dataloader = DataLoader_RNN(batch_size, user_1hop_file_train, item_1hop_file_train, target_file_train, feature_size, max_len, k)
            
            for batch_data in dataloader:
                if early_stop:
                    break

                loss = model.train(sess, batch_data, lr, reg_lambda)
                step += 1

                if step % eval_iter_num == 0:
                    train_loss, train_auc, _ = eval(model, model_type, sess, item_1hop_file_train, user_1hop_file_train, item_2hop_file_train, user_2hop_file_train, target_file_train, feature_size, max_len, k)
                    test_loss, test_auc, _ = eval(model, model_type, sess, item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test, feature_size, max_len, k)

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    train_aucs.append(train_auc)
                    test_aucs.append(test_auc)

                    print("STEP %d  LOSS_TRAIN: %.4f  LOSS_TEST: %.4f  AUC_TRAIN: %.4f  AUC_TEST: %.4f" % (step, train_loss, test_loss, train_auc, test_auc))
                    if test_aucs[-1] > max(test_aucs[:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}_{}'.format(model_type, train_num, train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/'.format(data_set, model_name)):
                            os.mkdir('save_model_{}/{}/'.format(data_set, model_name))
                        save_path = 'save_model_{}/{}/ckpt'.format(data_set, model_name)
                        model.save(sess, save_path)

                if len(test_losses) > 2 and epoch > 0:
                    if (test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3]):
                        early_stop = True

        # generate log
        logname = '{}_{}_{}_{}'.format(model_type, train_num, lr, reg_lambda)

        with open('logs_{}/{}'.format(data_set, logname), 'wb') as f:
            dump_tuple = (train_losses, test_losses, train_aucs, test_aucs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, logname), 'w') as f:
            f.write('Result Test AUC: {}\n'.format(max(test_aucs)))
            f.write('Result Test Logloss: {}\n'.format(test_losses[np.argmax(test_aucs)]))

        return max(test_aucs), test_losses[np.argmax(test_aucs)]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("PLEASE INPUT [MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    model_type = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    data_set = sys.argv[3]

    if data_set == 'taobao':
        item_1hop_file_train = '../../data/taobao/score-data/item-1hop-train.txt'
        user_1hop_file_train = '../../data/taobao/score-data/user-1hop-train.txt'
        item_2hop_file_train = '../../data/taobao/score-data/item-2hop-train.txt'
        user_2hop_file_train = '../../data/taobao/score-data/user-2hop-train.txt'
        target_file_train = '../../data/taobao/score-data/target-train.txt'
        
        item_1hop_file_test = '../../data/taobao/score-data/item-1hop-test.txt'
        user_1hop_file_test = '../../data/taobao/score-data/user-1hop-test.txt'
        item_2hop_file_test = '../../data/taobao/score-data/item-2hop-test.txt'
        user_2hop_file_test = '../../data/taobao/score-data/user-2hop-test.txt'
        target_file_test = '../../data/taobao/score-data/target-test.txt'

        # model parameter
        feature_size = 5163070 + 2 
        max_len_slice = 18
        max_len_point = 50 
        k = 5
        user_fnum = 1 
        item_fnum = 2
        eval_iter_num = 650
    
    elif data_set == 'amazon':
        item_1hop_file_train = '../../data/amazon/Books/score-data/item-1hop-train.txt'
        user_1hop_file_train = '../../data/amazon/Books/score-data/user-1hop-train.txt'
        item_2hop_file_train = '../../data/amazon/Books/score-data/item-2hop-train.txt'
        user_2hop_file_train = '../../data/amazon/Books/score-data/user-2hop-train.txt'
        target_file_train = '../../data/amazon/Books/score-data/target-train.txt'
        
        item_1hop_file_test = '../../data/amazon/Books/score-data/item-1hop-test.txt'
        user_1hop_file_test = '../../data/amazon/Books/score-data/user-1hop-test.txt'
        item_2hop_file_test = '../../data/amazon/Books/score-data/item-2hop-test.txt'
        user_2hop_file_test = '../../data/amazon/Books/score-data/user-2hop-test.txt'
        target_file_test = '../../data/amazon/Books/score-data/target-test.txt'

        # model parameter
        feature_size = 973218 + 2 
        max_len_slice = 19
        max_len_point = 30
        k = 3
        user_fnum = 1 
        item_fnum = 2
        eval_iter_num = 200

    elif data_set == 'douban':
        item_1hop_file_train = '../../data/douban/score-data/item-1hop-train.txt'
        user_1hop_file_train = '../../data/douban/score-data/user-1hop-train.txt'
        item_2hop_file_train = '../../data/douban/score-data/item-2hop-train.txt'
        user_2hop_file_train = '../../data/douban/score-data/user-2hop-train.txt'
        target_file_train = '../../data/douban/score-data/target-train.txt'
        
        item_1hop_file_test = '../../data/douban/score-data/item-1hop-test.txt'
        user_1hop_file_test = '../../data/douban/score-data/user-1hop-test.txt'
        item_2hop_file_test = '../../data/douban/score-data/item-2hop-test.txt'
        user_2hop_file_test = '../../data/douban/score-data/user-2hop-test.txt'
        target_file_test = '../../data/douban/score-data/target-test.txt'

        # model parameter
        feature_size = 973551 + 2 
        max_len_slice = 22
        max_len_point = 30
        k = 5
        user_fnum = 1 
        item_fnum = 2
        eval_iter_num = 1000

    ################################## training hyper params ##################################
    train_batch_sizes = [100]
    lrs = [1e-3]
    reg_lambdas = [0]

    for train_batch_size in train_batch_sizes:
        for lr in lrs:
            for reg_lambda in reg_lambdas:
                test_aucs = []
                test_loglosses = []
                for i in range(1):
                    test_auc, test_logloss = train(item_1hop_file_train, user_1hop_file_train, item_2hop_file_train, user_2hop_file_train, target_file_train,
                                                    item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test,
                                                    model_type, i, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_point, max_len_slice, k, user_fnum, item_fnum, lr, reg_lambda, data_set, eval_iter_num)
                    
                    restore(item_1hop_file_test, user_1hop_file_test, item_2hop_file_test, user_2hop_file_test, target_file_test,
                            model_type, i, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_point, max_len_slice, k, user_fnum, item_fnum, lr, reg_lambda, data_set)
                    
                    test_aucs.append(test_auc)
                    test_loglosses.append(test_logloss)

    #             if sum(test_aucs) / 5 > result_auc:
    #                 result_auc = sum(test_aucs) / 5
    #             if sum(test_loglosses) / 5 < result_logloss:
    #                 result_logloss = sum(test_loglosses) / 5
    # print("FINAL RESULT: AUC=%.4f\tLOGLOSS=%.4f" % (result_auc, result_logloss))
