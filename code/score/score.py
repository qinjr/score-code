import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

NEG_SAMPLE_NUM = 9

'''
Slice Based Models: SCORE, RRN, GCMC
'''
class SCOREBASE(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, neg_sample_num):
        # reset graph
        tf.reset_default_graph()

        self.obj_per_time_slice = obj_per_time_slice
        self.neg_sample_num = neg_sample_num

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_1hop_pos_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='user_1hop_pos_ph')
            self.user_2hop_pos_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='user_2hop_pos_ph')
            self.user_1hop_neg_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='user_1hop_neg_ph')
            self.user_2hop_neg_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='user_2hop_neg_ph')
            
            self.item_1hop_pos_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='item_1hop_pos_ph')
            self.item_2hop_pos_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='item_2hop_pos_ph')
            self.item_1hop_neg_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='item_1hop_neg_ph')
            self.item_2hop_neg_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='item_2hop_neg_ph')

            self.target_user_ph = tf.placeholder(tf.int32, [None,], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None,], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')
            self.length_ph = tf.placeholder(tf.int32, [None,], name='length_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # lambda and mu
            self.reg_lambda = tf.placeholder(tf.float32, [])
            self.mu = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
        
        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
            self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            self.emb_mtx = self.emb_mtx * self.emb_mtx_mask
            
            # user interaction set and co-interaction set
            self.user_1hop_pos = tf.nn.embedding_lookup(self.emb_mtx, self.user_1hop_pos_ph)
            self.user_2hop_pos = tf.nn.embedding_lookup(self.emb_mtx, self.user_2hop_pos_ph)
    
            self.user_1hop_neg = tf.nn.embedding_lookup(self.emb_mtx, self.user_1hop_neg_ph)
            self.user_2hop_neg = tf.nn.embedding_lookup(self.emb_mtx, self.user_2hop_neg_ph)
            
            # item interaction set and co-interaction set
            self.item_1hop_pos = tf.nn.embedding_lookup(self.emb_mtx, self.item_1hop_pos_ph)
            self.item_2hop_pos = tf.nn.embedding_lookup(self.emb_mtx, self.item_2hop_pos_ph)
            
            self.item_1hop_neg = tf.nn.embedding_lookup(self.emb_mtx, self.item_1hop_neg_ph)
            self.item_2hop_neg = tf.nn.embedding_lookup(self.emb_mtx, self.item_2hop_neg_ph)
            
            # target item and target user
            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            
    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1,])
    
    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
    
    def build_l2norm(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

    def build_train_step(self):
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
    
    def train(self, sess, batch_data, lr, reg_lambda, mu):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.user_1hop_pos_ph : batch_data[0],
                self.user_2hop_pos_ph : batch_data[1],
                self.item_1hop_pos_ph : batch_data[2],
                self.item_2hop_pos_ph : batch_data[3],
                self.user_1hop_neg_ph : batch_data[4],
                self.user_2hop_neg_ph : batch_data[5],
                self.item_1hop_neg_ph : batch_data[6],
                self.item_2hop_neg_ph : batch_data[7],
                self.target_user_ph : batch_data[8],
                self.target_item_ph : batch_data[9],
                self.label_ph : batch_data[10],
                self.length_ph : batch_data[11],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.reg_lambda : mu,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda, mu):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.user_1hop_pos_ph : batch_data[0],
                self.user_2hop_pos_ph : batch_data[1],
                self.item_1hop_pos_ph : batch_data[2],
                self.item_2hop_pos_ph : batch_data[3],
                self.user_1hop_neg_ph : batch_data[4],
                self.user_2hop_neg_ph : batch_data[5],
                self.item_1hop_neg_ph : batch_data[6],
                self.item_2hop_neg_ph : batch_data[7],
                self.target_user_ph : batch_data[8],
                self.target_item_ph : batch_data[9],
                self.label_ph : batch_data[10],
                self.length_ph : batch_data[11],
                self.reg_lambda : reg_lambda,
                self.reg_lambda : mu,
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))
    
    def co_attention(self, seq1, seq2):
        with tf.variable_scope('co-attention'):
            seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False, name='co_atten_dense_1', reuse=tf.AUTO_REUSE)
            seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], use_bias=False, name='co_atten_dense_2', reuse=tf.AUTO_REUSE)
            seq2 = tf.layers.dense(seq2, seq2.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False, name='co_atten_dense_3', reuse=tf.AUTO_REUSE)
        
        product = tf.matmul(seq1, tf.transpose(seq2, [0, 1, 3, 2]))

        seq1_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=3)), axis=3)
        seq2_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=2)), axis=3)

        seq1_result = tf.reduce_sum(seq1 * seq1_weights, axis=2) #[B, T, D]
        seq2_result = tf.reduce_sum(seq2 * seq2_weights, axis=2)

        return seq1_result, seq2_result, seq1_weights, seq2_weights

class SCORE(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(SCORE, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)
        # co-attention graph aggregator
        user_1hop_seq_pos, item_2hop_seq_pos, self.user_1hop_wei_pos, self.item_2hop_wei_pos = self.co_attention(self.user_1hop_pos, self.item_2hop_pos)
        user_2hop_seq_pos, item_1hop_seq_pos, self.user_2hop_wei_pos, self.item_1hop_wei_pos = self.co_attention(self.user_2hop_pos, self.item_1hop_pos)
        user_1hop_seq_neg, item_2hop_seq_neg, self.user_1hop_wei_neg, self.item_2hop_wei_neg = self.co_attention(self.user_1hop_neg, self.item_2hop_neg)
        user_2hop_seq_neg, item_1hop_seq_neg, self.user_2hop_wei_neg, self.item_1hop_wei_neg = self.co_attention(self.user_2hop_neg, self.item_1hop_neg)

        user_side_pos = user_1hop_seq_pos + user_2hop_seq_pos#tf.concat([user_1hop_seq_pos, user_2hop_seq_pos], axis=2)
        item_side_pos = item_1hop_seq_pos + item_2hop_seq_pos#tf.concat([item_1hop_seq_pos, item_2hop_seq_pos], axis=2)
        user_side_neg = user_1hop_seq_neg + user_2hop_seq_neg#tf.concat([user_1hop_seq_neg, user_2hop_seq_neg], axis=2)
        item_side_neg = item_1hop_seq_neg + item_2hop_seq_neg#tf.concat([item_1hop_seq_neg, item_2hop_seq_neg], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side_pos, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side_pos, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.loss += self.mu * -tf.reduce_mean(tf.log(1 - tf.sigmoid(tf.reduce_sum(user_side_pos * user_side_neg, axis=2))))
        self.loss += self.mu * -tf.reduce_mean(tf.log(1 - tf.sigmoid(tf.reduce_sum(item_side_pos * item_side_neg, axis=2))))
        