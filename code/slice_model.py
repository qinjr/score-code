import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

NEG_SAMPLE_NUM = 9

'''
Slice Based Models: SCORE, RRN, GCMC
'''
class SliceBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num):
        # reset graph
        tf.reset_default_graph()

        self.obj_per_time_slice = obj_per_time_slice
        self.neg_sample_num = neg_sample_num

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_1hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, item_fnum], name='user_1hop_ph')
            self.user_2hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, user_fnum], name='user_2hop_ph')
            self.item_1hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, user_fnum], name='item_1hop_ph')
            self.item_2hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, item_fnum], name='item_2hop_ph')

            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')
            self.length_ph = tf.placeholder(tf.int32, [None,], name='length_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
        
        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
            self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

            self.user_1hop = tf.nn.embedding_lookup(self.emb_mtx, self.user_1hop_ph)
            user_1hop_shape = self.user_1hop.get_shape().as_list()
            self.user_1hop = tf.reshape(self.user_1hop, [-1, user_1hop_shape[1], user_1hop_shape[2], user_1hop_shape[3] * user_1hop_shape[4]])

            self.user_2hop = tf.nn.embedding_lookup(self.emb_mtx, self.user_2hop_ph)
            user_2hop_shape = self.user_2hop.get_shape().as_list()
            self.user_2hop = tf.reshape(self.user_2hop, [-1, user_2hop_shape[1], user_2hop_shape[2], user_2hop_shape[3] * user_2hop_shape[4]])

            self.item_1hop = tf.nn.embedding_lookup(self.emb_mtx, self.item_1hop_ph)
            item_1hop_shape = self.item_1hop.get_shape().as_list()
            self.item_1hop = tf.reshape(self.item_1hop, [-1, item_1hop_shape[1], item_1hop_shape[2], item_1hop_shape[3] * item_1hop_shape[4]])

            self.item_2hop = tf.nn.embedding_lookup(self.emb_mtx, self.item_2hop_ph)
            item_2hop_shape = self.item_2hop.get_shape().as_list()
            self.item_2hop = tf.reshape(self.item_2hop, [-1, item_2hop_shape[1], item_2hop_shape[2], item_2hop_shape[3] * item_2hop_shape[4]])

            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
            target_item_shape = self.target_item.get_shape().as_list()
            self.target_item = tf.reshape(self.target_item, [-1, target_item_shape[1] * target_item_shape[2]])
            
            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            target_user_shape = self.target_user.get_shape().as_list()
            self.target_user = tf.reshape(self.target_user, [-1, target_user_shape[1] * target_user_shape[2]])
        
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
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_bprloss(self):
        self.pred_reshape = tf.reshape(self.y_pred, [-1, self.neg_sample_num + 1])
        self.pred_pos = tf.tile(tf.expand_dims(self.pred_reshape[:, 0], 1), [1, self.neg_num])
        self.pred_neg = self.pred_reshape[:, 1:]
        self.loss = tf.reduce_mean(tf.log(tf.nn.sigmoid(self.pred_pos - self.pred_neg)))
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
    
    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.user_1hop_ph : batch_data[0],
                self.user_2hop_ph : batch_data[1],
                self.item_1hop_ph : batch_data[2],
                self.item_2hop_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.length_ph : batch_data[7],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.user_1hop_ph : batch_data[0],
                self.user_2hop_ph : batch_data[1],
                self.item_1hop_ph : batch_data[2],
                self.item_2hop_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.length_ph : batch_data[7],
                self.reg_lambda : reg_lambda,
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss
    
    def summary(self, sess, batch_data, reg_lambda):
        summary = sess.run(self.merged_summary, feed_dict = {
                self.user_1hop_ph : batch_data[0],
                self.user_2hop_ph : batch_data[1],
                self.item_1hop_ph : batch_data[2],
                self.item_2hop_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.length_ph : batch_data[7],
                self.reg_lambda : reg_lambda,
            })
        return summary
    
    def get_co_attention(self, sess, batch_data):
        user_1hop_wei, user_2hop_wei, item_1hop_wei, item_2hop_wei = sess.run([self.user_1hop_wei, self.user_2hop_wei, self.item_1hop_wei, self.item_2hop_wei], feed_dict = {
                self.user_1hop_ph : batch_data[0],
                self.user_2hop_ph : batch_data[1],
                self.item_1hop_ph : batch_data[2],
                self.item_2hop_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.length_ph : batch_data[7],
            })
        return user_1hop_wei, user_2hop_wei, item_1hop_wei, item_2hop_wei

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

class SCORE(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(SCORE, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)
        user_1hop_seq, item_2hop_seq, self.user_1hop_wei, self.item_2hop_wei = self.co_attention_v1(self.user_1hop, self.item_2hop)
        user_2hop_seq, item_1hop_seq, self.user_2hop_wei, self.item_1hop_wei = self.co_attention_v1(self.user_2hop, self.item_1hop)
        # summary node
        # tf.summary.histogram('user_1hop_wei', user_1hop_wei)
        # tf.summary.histogram('user_2hop_wei', user_2hop_wei)
        # tf.summary.histogram('item_1hop_wei', item_1hop_wei)
        # tf.summary.histogram('item_2hop_wei', item_2hop_wei)

        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()
        # merged summary
        # self.merged_summary = tf.summary.merge_all()

    def co_attention_v1(self, seq1, seq2):
        seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], use_bias=False)
        seq2 = tf.layers.dense(seq2, seq2.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        product = tf.matmul(seq1, tf.transpose(seq2, [0, 1, 3, 2]))

        seq1_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=3)), axis=3)
        seq2_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=2)), axis=3)

        seq1_result = tf.reduce_sum(seq1 * seq1_weights, axis=2) #[B, T, D]
        seq2_result = tf.reduce_sum(seq2 * seq2_weights, axis=2)

        return seq1_result, seq2_result, seq1_weights, seq2_weights
    

class SCORE_1HOP(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(SCORE_1HOP, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)
        user_1hop_seq, item_2hop_seq, self.user_1hop_wei, self.item_2hop_wei = self.co_attention_v1(self.user_1hop, self.item_2hop)
        user_2hop_seq, item_1hop_seq, self.user_2hop_wei, self.item_1hop_wei = self.co_attention_v1(self.user_2hop, self.item_1hop)
        # summary node
        # tf.summary.histogram('user_1hop_wei', user_1hop_wei)
        # tf.summary.histogram('user_2hop_wei', user_2hop_wei)
        # tf.summary.histogram('item_1hop_wei', item_1hop_wei)
        # tf.summary.histogram('item_2hop_wei', item_2hop_wei)

        user_side = tf.concat([user_1hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()
        # merged summary
        # self.merged_summary = tf.summary.merge_all()

class SCORE_v2(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(SCORE_v2, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)
        user_1hop_gat = self.gat(self.user_1hop, self.user_1hop, self.target_item)
        item_1hop_gat = self.gat(self.item_1hop, self.item_1hop, self.target_user)

        user_1hop_seq, item_2hop_seq, user_1hop_wei, item_2hop_wei = self.co_attention_v1(self.user_1hop, self.item_2hop)
        user_2hop_seq, item_1hop_seq, user_2hop_wei, item_1hop_wei = self.co_attention_v1(self.user_2hop, self.item_1hop)
        
        user_side = tf.concat([user_1hop_gat, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_gat, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()
    
    def lrelu(self, x, alpha=0.2):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def gat(self, key, value, query):
        # key/value: [B, T, K, D], query: [B, D]
        key_shape = key.get_shape().as_list()
        query = tf.expand_dims(tf.expand_dims(query, 1), 2)
        query = tf.tile(query, [1, key_shape[1], key_shape[2], 1]) #[B, T, K, D]
        query_key_concat = tf.concat([query, key], axis = 3)
        atten = tf.layers.dense(query_key_concat, 1, activation=None, use_bias=False) #[B, T, K, 1]
        atten = self.lrelu(atten)
        atten = tf.reshape(atten, [-1, key_shape[1], key_shape[2]]) #[B, T, K]
        atten = tf.expand_dims(tf.nn.softmax(atten), 3) #[B, T, K, 1]
        res = tf.reduce_sum(atten * value, axis=2)
        return res

    def co_attention_v1(self, seq1, seq2):
        seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], use_bias=False)
        seq2 = tf.layers.dense(seq2, seq2.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        product = tf.matmul(seq1, tf.transpose(seq2, [0, 1, 3, 2]))

        seq1_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=3)) * seq1.get_shape().as_list()[2], axis=3)
        seq2_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=2)) * seq1.get_shape().as_list()[2], axis=3)

        seq1_result = tf.reduce_sum(seq1 * seq1_weights, axis=2) #[B, T, D]
        seq2_result = tf.reduce_sum(seq2 * seq2_weights, axis=2)

        return seq1_result, seq2_result, seq1_weights, seq2_weights

class RRN(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(RRN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)
        user_side = tf.reduce_sum(self.user_1hop, axis=2)
        item_side = tf.reduce_sum(self.item_1hop, axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class RRN_AVG(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(RRN_AVG, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)
        user_side = tf.reduce_mean(self.user_1hop, axis=2)
        item_side = tf.reduce_mean(self.item_1hop, axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class GCMC(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(GCMC, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum, neg_sample_num)

        user_1hop_li = tf.layers.dense(self.user_1hop, self.user_1hop.get_shape().as_list()[-1], activation=None, use_bias=False)
        item_1hop_li = tf.layers.dense(self.item_1hop, self.item_1hop.get_shape().as_list()[-1], activation=None, use_bias=False)

        # sum pooling
        user_1hop_seq_sum = tf.nn.relu(tf.reduce_sum(user_1hop_li, axis=2))
        item_1hop_seq_sum = tf.nn.relu(tf.reduce_sum(item_1hop_li, axis=2))

        user_1hop_seq = tf.layers.dense(user_1hop_seq_sum, user_1hop_seq_sum.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        item_1hop_seq = tf.layers.dense(item_1hop_seq_sum, item_1hop_seq_sum.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_1hop_seq, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru1')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_1hop_seq, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru2')
        
        # inp = tf.concat([item_seq_final_state, user_seq_final_state, self.target_user, , self.target_item], axis=1) #, self.target_user

        # pred
        self.y_pred_pos = tf.exp(tf.reduce_sum(tf.layers.dense(item_side_final_state, hidden_size, use_bias=False) * user_side_final_state, axis=1))
        self.y_pred_neg = tf.exp(tf.reduce_sum(tf.layers.dense(item_side_final_state, hidden_size, use_bias=False) * user_side_final_state, axis=1))
        self.y_pred = self.y_pred_pos / (self.y_pred_pos + self.y_pred_neg)

        self.build_logloss()


