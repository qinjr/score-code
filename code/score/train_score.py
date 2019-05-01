import os
import tensorflow as tf
import sys
from graph_loader import *
from score import *
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
TRAIN_NEG_SAMPLE_NUM = 1
TEST_NEG_SAMPLE_NUM = 99

WORKER_N = 5

# for CCMR
OBJ_PER_TIME_SLICE_CCMR = 10
TIME_SLICE_NUM_CCMR = 41
START_TIME_CCMR = 0
FEAT_SIZE_CCMR = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1)
DATA_DIR_CCMR = '../../../score-data/CCMR/feateng/'
USER_PER_COLLECTION_CCMR = 1000
ITEM_PER_COLLECTION_CCMR = 100
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129

# for Taobao
OBJ_PER_TIME_SLICE_Taobao = 10
TIME_SLICE_NUM_Taobao = 18
START_TIME_Taobao = 0
FEAT_SIZE_Taobao = 1 + 984105 + 4067842 + 9405
DATA_DIR_Taobao = '../../../score-data/Taobao/feateng/'
USER_PER_COLLECTION_Taobao = 500
ITEM_PER_COLLECTION_Taobao = 500
USER_NUM_Taobao = 984105
ITEM_NUM_Taobao = 4067842

# for Tmall
OBJ_PER_TIME_SLICE_Tmall = 10
TIME_SLICE_NUM_Tmall = 13
START_TIME_Tmall = 0
FEAT_SIZE_Tmall = 1529672
DATA_DIR_Tmall = '../../../score-data/Tmall/feateng/'
USER_PER_COLLECTION_Tmall = 200
ITEM_PER_COLLECTION_Tmall = 250
USER_NUM_Tmall = 424170
ITEM_NUM_Tmall = 1090390

def restore(data_set, target_file_test, graph_handler_params, start_time,
        pred_time_test, model_type, train_batch_size, feature_size, eb_dim, 
        hidden_size, max_time_len, obj_per_time_slice, lr, reg_lambda, mu):
    print('restore begin')
    graph_handler_params = graph_handler_params
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_V2':
        model = SCORE_V2(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_CONCAT':
        model = SCORE_CONCAT(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_ATT':
        model = SCORE_ATT(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_V3':
        model = SCORE_V3(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda, mu)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        _, _, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss, auxloss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda, mu)
        # p = 1. / (1 + TEST_NEG_SAMPLE_NUM)
        # rig = 1 -(logloss / -(p * math.log(p) + (1 - p) * math.log(1 - p)))
        print('RESTORE, LOSS TEST: %.4f  AUXLOSS TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f' % (loss, auxloss, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr))

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

def eval(model, sess, graph_handler_params, target_file, start_time, pred_time, 
        reg_lambda, mu):
    preds = []
    labels = []
    target_iids = []
    losses = []
    auxlosses = []
    graph_loader = GraphLoader(graph_handler_params, EVAL_BATCH_SIZE, target_file, start_time, pred_time, WORKER_N, TEST_NEG_SAMPLE_NUM)
    t = time.time()
    for batch_data in graph_loader:
        pred, label, loss, auxloss = model.eval(sess, batch_data, reg_lambda, mu)
        preds += pred
        labels += label
        losses.append(loss)
        auxlosses.append(auxloss)
        target_iids += np.array(batch_data[5]).tolist()
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    auxloss = sum(auxlosses) / len(auxlosses)
    # if mode == 'train':
    #     ndcg = get_ndcg(preds, target_iids)
    #     print("EVAL TIME: %.4fs" % (time.time() - t))
    #     return logloss, auc, ndcg, loss
    # elif mode == 'restore':
    ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr = get_ranking_quality(preds, target_iids)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return logloss, auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss, auxloss
    
def train(data_set, target_file_train, target_file_test, graph_handler_params, start_time,
        pred_time_train, pred_time_test, model_type, train_batch_size, feature_size, 
        eb_dim, hidden_size, max_time_len, obj_per_time_slice, lr, reg_lambda, mu, dataset_size):
    graph_handler_params = graph_handler_params
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_V2':
        model = SCORE_V2(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_CONCAT':
        model = SCORE_CONCAT(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_ATT':
        model = SCORE_ATT(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
    elif model_type == 'SCORE_V3':
        model = SCORE_V3(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
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

        # test_loglosses = []
        # test_aucs = []
        test_ndcgs_5 = []
        test_ndcgs_10 = []
        test_hrs_1 = []
        test_hrs_5 = []
        test_hrs_10 = []
        test_mrrs = []
        test_losses = []
        test_auxlosses = []

        # before training process
        step = 0
        _, _, test_ndcg_5, test_ndcg_10, test_hr_1, test_hr_5, test_hr_10, test_mrr, test_loss, test_auxloss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda, mu)
        # test_loglosses.append(test_logloss)
        # test_aucs.append(test_auc)
        test_ndcgs_5.append(test_ndcg_5)
        test_ndcgs_10.append(test_ndcg_10)
        test_hrs_1.append(test_hr_1)
        test_hrs_5.append(test_hr_5)
        test_hrs_10.append(test_hr_10)
        test_mrrs.append(test_mrr)
        test_losses.append(test_loss)
        test_auxlosses.append(test_auxloss)

        print("STEP %d  LOSS TRAIN: NULL  LOSS TEST: %.4f  AUXLOSS TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f" % (step, test_loss, test_auxloss, test_ndcg_5, test_ndcg_10, test_hr_1, test_hr_5, test_hr_10, test_mrr))
        early_stop = False
        eval_iter_num = (dataset_size // 5) // (train_batch_size / (1 + TRAIN_NEG_SAMPLE_NUM))
        # begin training process
        for epoch in range(10):
            if early_stop:
                break
            graph_loader = GraphLoader(graph_handler_params, train_batch_size, target_file_train, start_time, pred_time_train, WORKER_N, TRAIN_NEG_SAMPLE_NUM)
            for batch_data in graph_loader:
                if early_stop:
                    break

                loss = model.train(sess, batch_data, lr, reg_lambda, mu)
                step += 1
                train_losses_step.append(loss)
                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []

                    # test_logloss, test_auc, test_ndcg, test_loss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda)
                    _, _, test_ndcg_5, test_ndcg_10, test_hr_1, test_hr_5, test_hr_10, test_mrr, test_loss, test_auxloss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda, mu)
                    # test_loglosses.append(test_logloss)
                    # test_aucs.append(test_auc)
                    test_ndcgs_5.append(test_ndcg_5)
                    test_ndcgs_10.append(test_ndcg_10)
                    test_hrs_1.append(test_hr_1)
                    test_hrs_5.append(test_hr_5)
                    test_hrs_10.append(test_hr_10)
                    test_mrrs.append(test_mrr)
                    test_losses.append(test_loss)
                    test_auxlosses.append(test_auxloss)
                    
                    print("STEP %d  LOSS TRAIN: %.4f  LOSS TEST: %.4f  AUXLOSS TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f" % (step, train_loss, test_loss, test_auxloss, test_ndcg_5, test_ndcg_10, test_hr_1, test_hr_5, test_hr_10, test_mrr))
                    if test_mrrs[-1] > min(test_mrrs[:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda, mu)
                        if not os.path.exists('save_model_{}/{}/'.format(data_set, model_name)):
                            os.makedirs('save_model_{}/{}/'.format(data_set, model_name))
                        save_path = 'save_model_{}/{}/ckpt'.format(data_set, model_name)
                        model.save(sess, save_path)
                    
                    if len(test_mrrs) > 2 and epoch > 0:
                        if (test_mrrs[-1] < test_mrrs[-2] and test_mrrs[-2] < test_mrrs[-3]):
                            early_stop = True
                        if (test_mrrs[-1] - test_mrrs[-2]) <= 0.001 and (test_mrrs[-2] - test_mrrs[-3]) <= 0.001:
                            early_stop = True
        # generate log
        if not os.path.exists('logs_{}/'.format(data_set)):
            os.makedirs('logs_{}/'.format(data_set))
        model_name = '{}_{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda, mu)

        with open('logs_{}/{}.pkl'.format(data_set, model_name), 'wb') as f:
            dump_tuple = (train_losses, test_losses, test_ndcg_5, test_ndcg_10, test_hr_1, test_hr_5, test_hr_10, test_mrrs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, model_name), 'w') as f:
            index = np.argmax(test_mrrs)
            f.write('Result Test NDCG@5: {}\n'.format(test_ndcgs_5[index]))
            f.write('Result Test NDCG@10: {}\n'.format(test_ndcgs_10[index]))
            f.write('Result Test HR@1: {}\n'.format(test_hrs_1[index]))
            f.write('Result Test HR@5: {}\n'.format(test_hrs_5[index]))
            f.write('Result Test HR@10: {}\n'.format(test_hrs_10[index]))
            f.write('Result Test MRR: {}\n'.format(test_mrrs[index]))
        return 

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
                                USER_NUM_CCMR, ITEM_NUM_CCMR, START_TIME_CCMR, \
                                USER_PER_COLLECTION_CCMR, \
                                ITEM_PER_COLLECTION_CCMR, 'is']
        target_file_train = DATA_DIR_CCMR + 'target_39_hot_sample.txt'
        target_file_test = DATA_DIR_CCMR + 'target_40_hot_sample.txt'
        start_time = START_TIME_CCMR
        pred_time_train = 39
        pred_time_test = 40
        # model parameter
        feature_size = FEAT_SIZE_CCMR
        max_time_len = TIME_SLICE_NUM_CCMR - START_TIME_CCMR - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_CCMR
        dataset_size = 262090
    elif data_set == 'taobao':
        # graph loader
        graph_handler_params = [TIME_SLICE_NUM_Taobao, 'taobao_2hop', OBJ_PER_TIME_SLICE_Taobao, \
                                USER_NUM_Taobao, ITEM_NUM_Taobao, START_TIME_Taobao, \
                                USER_PER_COLLECTION_Taobao, ITEM_PER_COLLECTION_Taobao, 'is']
        target_file_train = DATA_DIR_Taobao + 'target_16_hot_train_sample.txt'
        target_file_test = DATA_DIR_Taobao + 'target_17_hot_test_sample.txt'
        start_time = START_TIME_Taobao
        pred_time_train = 16
        pred_time_test = 17
        # model parameter
        feature_size = FEAT_SIZE_Taobao
        max_time_len = TIME_SLICE_NUM_Taobao - START_TIME_Taobao - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_Taobao
        dataset_size = 262239
    elif data_set == 'tmall':
        # graph loader
        graph_handler_params = [TIME_SLICE_NUM_Tmall, 'tmall_2hop', OBJ_PER_TIME_SLICE_Tmall, \
                                USER_NUM_Tmall, ITEM_NUM_Tmall, START_TIME_Tmall, \
                                USER_PER_COLLECTION_Tmall, ITEM_PER_COLLECTION_Tmall, 'is']
        target_file_train = DATA_DIR_Tmall + 'target_10_hot_sample.txt'
        target_file_test = DATA_DIR_Tmall + 'target_11_hot_sample.txt'
        start_time = START_TIME_Tmall
        pred_time_train = 10
        pred_time_test = 11
        # model parameter
        feature_size = FEAT_SIZE_Tmall
        max_time_len = TIME_SLICE_NUM_Tmall - START_TIME_Tmall - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_Tmall
        dataset_size = 228213
    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    reg_lambda = 1e-3
    mu = 1e-2
    hyper_paras = [(100, 5e-4), (200, 1e-3)]

    for hyper in hyper_paras:
        train_batch_size, lr = hyper
        train(data_set, target_file_train, target_file_test, graph_handler_params, start_time,
                pred_time_train, pred_time_test, model_type, train_batch_size, feature_size, 
                EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, obj_per_time_slice, lr, reg_lambda, mu, dataset_size)
        
        restore(data_set, target_file_test, graph_handler_params, start_time,
                pred_time_test, model_type, train_batch_size, feature_size, 
                EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, obj_per_time_slice, 
                lr, reg_lambda, mu)