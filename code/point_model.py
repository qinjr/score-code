import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

NEG_SAMPLE_NUM = 9

'''
Point Based Models: GRU4Rec
'''
class PointBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                user_fnum, item_fnum, neg_sample_num):
        # reset graph
        tf.reset_default_graph()

        self.neg_sample_num = neg_sample_num

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_seq_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum], name='user_seq_ph')
            self.user_seq_length_ph = tf.placeholder(tf.int32, [None,], name='user_seq_length_ph')
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

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

            self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_seq_ph)
            user_seq_shape = self.user_seq.get_shape().as_list()
            self.user_seq = tf.reshape(self.user_seq, [-1, user_seq_shape[1], user_seq_shape[2] * user_seq_shape[3]])

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
                self.user_seq_ph : batch_data[0],
                self.user_seq_length_ph : batch_data[1],
                self.target_user_ph : batch_data[2],
                self.target_item_ph : batch_data[3],
                self.label_ph : batch_data[4],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.user_seq_ph : batch_data[0],
                self.user_seq_length_ph : batch_data[1],
                self.target_user_ph : batch_data[2],
                self.target_item_ph : batch_data[3],
                self.label_ph : batch_data[4],
                self.reg_lambda : reg_lambda,
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

class GRU4Rec(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                        user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(GRU4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, 
                                    user_fnum, item_fnum, neg_sample_num)

        # GRU
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_length_ph, dtype=tf.float32, scope='gru1')
        
        inp = tf.concat([user_seq_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class Caser(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                        user_fnum, item_fnum, neg_sample_num = NEG_SAMPLE_NUM):
        super(Caser, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, 
                                    user_fnum, item_fnum, neg_sample_num)
        
        with tf.name_scope('user_seq_cnn'):
            # horizontal filters
            filters_user = 32
            h_kernel_size_user = [8, item_fnum * eb_dim]
            v_kernel_size_user = [self.user_seq.get_shape().as_list()[1], 1]

            self.user_seq = tf.expand_dims(self.user_seq, 3)
            conv1 = tf.layers.conv2d(self.user_seq, filters_user, h_kernel_size_user)
            max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
            user_hori_out = tf.reshape(max1, [-1, filters_user]) #[B, F]

            # vertical
            conv2 = tf.layers.conv2d(self.user_seq, filters_user, v_kernel_size_user)
            conv2 = tf.reshape(conv2, [-1, item_fnum * eb_dim, filters_user])
            user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, item_fnum * eb_dim])

            inp = tf.concat([user_hori_out, user_vert_out, self.target_item, self.target_user], axis=1)

        # fully connected layer
        self.build_fc_net(inp)
        self.build_logloss()

    
