import os
import tensorflow as tf
import sys
from graph_loader import *
from model import *
from sklearn.metrics import *
import random
import time
import numpy as np
import pickle as pkl
import math

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2
EVAL_BATCH_SIZE = 100
NEG_SAMPLE_NUM = 9

# for CCMR
OBJ_PER_TIME_SLICE_CCMR = 10
TIME_SLICE_NUM_CCMR = 41
START_TIME_CCMR = 30
FEAT_SIZE_CCMR = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1)
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'

def restore(data_set, target_file_test, graph_handler_params, start_time,
        pred_time_test, user_feat_dict_file, item_feat_dict_file,
        model_type, train_batch_size, feature_size, eb_dim, hidden_size, max_time_len, 
        obj_per_time_slice, user_fnum, item_fnum, lr, reg_lambda):
    print('restore begin')
    graph_handler_params = graph_handler_params
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        graph_handler_params.append('sample')
    elif model_type == 'RRN': 
        model = RRN(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        graph_handler_params.append('fix')
    elif model_type == 'GCMC': 
        model = GCMC(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        graph_handler_params.append('fix')
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        logloss, auc, ndcg, loss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda, user_feat_dict_file, item_feat_dict_file)
        print('RESTORE, LOSS TEST: %.4f  LOGLOSS TEST: %.4f  AUC TEST: %.4f  NDCG@10 TEST: %.4f' % (loss, logloss, auc, ndcg))

def getNDCG(ranklist, target_item):
    for i in range(len(ranklist)):
        if ranklist[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def get_ndcg(preds, target_iids):
    preds = np.array(preds).reshape(-1, NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, NEG_SAMPLE_NUM + 1)[:,0].flatten().tolist()
    ndcg_val = []
    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        ndcg_val.append(getNDCG(ranklist, pos_iids[i]))
    return np.mean(ndcg_val)

def eval(model, sess, graph_handler_params, target_file, start_time, pred_time, reg_lambda, 
        user_feat_dict_file, item_feat_dict_file):
    preds = []
    labels = []
    target_iids = []
    losses = []

    graph_loader = GraphLoader(graph_handler_params, EVAL_BATCH_SIZE, target_file, start_time, pred_time, user_feat_dict_file, item_feat_dict_file)
    t = time.time()
    for batch_data in graph_loader:
        pred, label, loss = model.eval(sess, batch_data, reg_lambda)
        preds += pred
        labels += label
        losses.append(loss)
        target_iids += np.array(batch_data[5])[:,0].tolist()
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    ndcg = get_ndcg(preds, target_iids)
    loss = sum(losses) / len(losses)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return logloss, auc, ndcg, loss

def train(data_set, target_file_train, target_file_test, graph_handler_params, start_time,
        pred_time_train, pred_time_test, user_feat_dict_file, item_feat_dict_file,
        model_type, train_batch_size, feature_size, eb_dim, hidden_size, max_time_len, 
        obj_per_time_slice, user_fnum, item_fnum, lr, reg_lambda, eval_iter_num):
    graph_handler_params = graph_handler_params
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        graph_handler_params.append('sample')
    elif model_type == 'RRN': 
        model = RRN(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        graph_handler_params.append('fix')
    elif model_type == 'GCMC': 
        model = GCMC(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        graph_handler_params.append('fix')
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
        test_logloss, test_auc, test_ndcg, test_loss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda, user_feat_dict_file, item_feat_dict_file)
        test_loglosses.append(test_logloss)
        test_aucs.append(test_auc)
        test_ndcgs.append(test_ndcg)
        test_losses.append(test_loss)

        print("STEP %d LOSS TRAIN: NaN  LOSS TEST: %.4f  LOGLOSS TEST: %.4f  AUC TEST: %.4f  NDCG@10 TEST: %.4f" % (step, test_loss, test_logloss, test_auc, test_ndcg))
        early_stop = False

        # begin training process
        for epoch in range(3):
            if early_stop:
                break
            graph_loader = GraphLoader(graph_handler_params, train_batch_size, target_file_train, start_time, pred_time_train, user_feat_dict_file, item_feat_dict_file)
            for batch_data in graph_loader:
                if early_stop:
                    break

                loss = model.train(sess, batch_data, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []
                    test_logloss, test_auc, test_ndcg, test_loss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda, user_feat_dict_file, item_feat_dict_file)

                    test_loglosses.append(test_logloss)
                    test_aucs.append(test_auc)
                    test_ndcgs.append(test_ndcg)
                    test_losses.append(test_loss)

                    print("STEP %d  LOSS TRAIN: %.4f  LOSS TEST: %.4f  LOGLOSS TEST: %.4f  AUC TEST: %.4f  NDCG@10 TEST: %.4f" % (step, train_loss, test_loss, test_logloss, test_auc, test_ndcg))
                    if test_aucs[-1] > max(test_aucs[:-1]):
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
        logname = '{}_{}_{}.pkl'.format(model_type, lr, reg_lambda)

        with open('logs_{}/{}'.format(data_set, logname), 'wb') as f:
            dump_tuple = (train_losses, test_losses, test_loglosses, test_aucs, test_ndcgs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, logname), 'w') as f:
            f.write('Result Test AUC: {}\n'.format(max(test_aucs)))
            f.write('Result Test Logloss: {}\n'.format(test_loglosses[np.argmax(test_aucs)]))
            f.write('Result Test NDCG@10: {}\n'.format(test_ndcgs[np.argmax(test_aucs)]))

        return max(test_aucs), test_loglosses[np.argmax(test_aucs)], test_ndcgs[np.argmax(test_aucs)]

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    model_type = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    data_set = sys.argv[3]

    if data_set == 'ccmr':
        # graph loader
        graph_handler_params = [TIME_SLICE_NUM_CCMR, 'ccmr_2hop', OBJ_PER_TIME_SLICE_CCMR, \
                                USER_NUM_CCMR, ITEM_NUM_CCMR, 1, 5, START_TIME_CCMR, None, \
                                DATA_DIR_CCMR + 'remap_movie_info_dict.pkl']
        target_file_train = DATA_DIR_CCMR + 'target_train.txt'#'target_train_splitbyuser_41.txt'#
        target_file_test = DATA_DIR_CCMR + 'target_test_sample.txt'#'target_test_splitbyuser_41.txt'#
        start_time = START_TIME_CCMR
        pred_time_train = 39
        pred_time_test = 40
        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_CCMR + 'remap_movie_info_dict.pkl'
        # model parameter
        feature_size = FEAT_SIZE_CCMR
        max_time_len = TIME_SLICE_NUM_CCMR - START_TIME_CCMR - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_CCMR
        user_fnum = 1 
        item_fnum = 5
        eval_iter_num = 6500#4000
    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    train_batch_sizes = [100]
    lrs = [1e-3]
    reg_lambdas = [1e-3]

    for train_batch_size in train_batch_sizes:
        for lr in lrs:
            for reg_lambda in reg_lambdas:
                test_auc, test_logloss, test_ndcg = train(data_set, target_file_train, target_file_test, graph_handler_params, start_time,
                                                pred_time_train, pred_time_test, user_feat_dict_file, item_feat_dict_file,
                                                model_type, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, 
                                                obj_per_time_slice, user_fnum, item_fnum, lr, reg_lambda, eval_iter_num)
                
                restore(data_set, target_file_test, graph_handler_params, start_time,
                        pred_time_test, user_feat_dict_file, item_feat_dict_file,
                        model_type, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, 
                        obj_per_time_slice, user_fnum, item_fnum, lr, reg_lambda)