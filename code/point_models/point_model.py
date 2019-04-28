import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np


'''
Point Based Models: GRU4Rec
'''
class PointBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len):
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_seq_ph = tf.placeholder(tf.int32, [None, max_time_len], name='user_seq_ph')
            self.user_seq_length_ph = tf.placeholder(tf.int32, [None,], name='user_seq_length_ph')
            self.target_user_ph = tf.placeholder(tf.int32, [None,], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None,], name='target_item_ph')
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
            # self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            # self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            # self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

            self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_seq_ph)
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
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len):
        super(GRU4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_time_len)

        # GRU
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_length_ph, dtype=tf.float32, scope='gru1')
        
        inp = tf.concat([user_seq_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class Caser(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len):
        super(Caser, self).__init__(feature_size, eb_dim, hidden_size, max_time_len)
        
        with tf.name_scope('user_seq_cnn'):
            # horizontal filters
            filters_user = 32
            h_kernel_size_user = [8, eb_dim]
            v_kernel_size_user = [self.user_seq.get_shape().as_list()[1], 1]

            self.user_seq = tf.expand_dims(self.user_seq, 3)
            conv1 = tf.layers.conv2d(self.user_seq, filters_user, h_kernel_size_user)
            max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
            user_hori_out = tf.reshape(max1, [-1, filters_user]) #[B, F]

            # vertical
            conv2 = tf.layers.conv2d(self.user_seq, filters_user, v_kernel_size_user)
            conv2 = tf.reshape(conv2, [-1, eb_dim, filters_user])
            user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, eb_dim])

            inp = tf.concat([user_hori_out, user_vert_out, self.target_item, self.target_user], axis=1)

        # fully connected layer
        self.build_fc_net(inp)
        self.build_logloss()

class ARNN(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len):
        super(ARNN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len)
        self.user_seq_mask = tf.sequence_mask(self.user_seq_length_ph, tf.shape(self.user_seq)[1], dtype=tf.float32) # [B, T]
        self.user_seq_mask = tf.expand_dims(self.user_seq_mask, -1) # [B, T, 1]
        with tf.name_scope('user_seq_gru'):
            user_seq_hidden_out, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_length_ph, dtype=tf.float32, scope='gru1')
        with tf.name_scope('user_seq_atten'):
            user_seq_final_state, _, __ = self.attention(user_seq_hidden_out, user_seq_hidden_out, self.target_item, self.user_seq_mask)

        inp = tf.concat([user_seq_final_state, self.target_item, self.target_user], axis=1)
        # fully connected layer
        self.build_fc_net(inp)
        self.build_logloss()
    
    def attention(self, key, value, query, mask):
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1]) # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis = -1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None) #[B, T, 1]

        mask = tf.equal(mask, tf.ones_like(mask)) #[B, T, 1]
        paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.reshape(tf.where(mask, fc3, paddings), [-1, max_len])) #[B, T]
        
        atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, atten_output, score

class SVDpp(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len):
        super(SVDpp, self).__init__(feature_size, eb_dim, hidden_size, max_time_len)
        # SVDFeature
        # with tf.name_scope('user_feature_rep'):
        #     self.user_feat_w_list = []
        #     for i in range(user_fnum):
        #         self.user_feat_w_list.append(tf.get_variable('user_feat_w_%d'%i, [], initializer=tf.truncated_normal_initializer))
        #     self.target_user_rep = self.target_user[:, :eb_dim] * self.user_feat_w_list[0]
        #     for i in range(1, user_fnum):
        #         self.target_user_rep += self.target_user[:,i*eb_dim:(i+1)*eb_dim] * self.user_feat_w_list[i]

        # with tf.name_scope('item_feature_rep'):
        #     self.item_feat_w_list = []
        #     for i in range(item_fnum):
        #         self.item_feat_w_list.append(tf.get_variable('item_feat_w_%d'%i, [], initializer=tf.truncated_normal_initializer))
        #     self.target_item_rep = self.target_item[:, :eb_dim] * self.item_feat_w_list[0]
        #     self.user_seq_rep = self.user_seq[:, :, :eb_dim] * self.item_feat_w_list[0]
        #     for i in range(1, item_fnum):
        #         self.target_item_rep += self.target_item[:,i*eb_dim:(i+1)*eb_dim] * self.item_feat_w_list[i]
        #         self.user_seq_rep += self.user_seq[:, :, i*eb_dim:(i+1)*eb_dim] * self.item_feat_w_list[i]
        
        # user and item bias
        with tf.name_scope('b'):
            self.item_user_bias = tf.get_variable('item_b', [feature_size, 1])
        self.target_user_rep = self.target_user[:, :eb_dim]
        self.target_item_rep = self.target_item[:, :eb_dim]
        self.user_seq_rep = self.user_seq[:, :, :eb_dim]
        # prediction
        self.user_seq_mask = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32), 2)
        self.user_seq_rep = self.user_seq_rep * self.user_seq_mask
        self.neighbor = tf.reduce_sum(self.user_seq_rep, axis=1)
        # self.norm_neighbor = self.neighbor / tf.sqrt(tf.expand_dims(tf.norm(self.user_seq_rep, 1, (1, 2)), 1))

        self.latent_score = tf.reduce_sum(self.target_item_rep * (self.target_user_rep + self.neighbor), 1)
        self.user_bias = tf.reshape(tf.nn.embedding_lookup(self.item_user_bias, self.target_user_ph), [-1,])
        self.item_bias = tf.reshape(tf.nn.embedding_lookup(self.item_user_bias, self.target_item_ph), [-1,])
        # self.average = 0.5
        self.y_pred = tf.nn.sigmoid(self.user_bias + self.item_bias + self.latent_score)
        
        self.build_logloss()
    
