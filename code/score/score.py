import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

'''
SCOREBASE Models: SCORE
'''
class SCOREBASE(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice):
        # reset graph
        tf.reset_default_graph()

        self.obj_per_time_slice = obj_per_time_slice

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_1hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='user_1hop_ph')
            self.user_2hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='user_2hop_ph')
            
            self.item_1hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='item_1hop_ph')
            self.item_2hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice], name='item_2hop_ph')
            
            self.target_user_ph = tf.placeholder(tf.int32, [None,], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None,], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')
            self.length_ph = tf.placeholder(tf.int32, [None,], name='length_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # regularization term
            self.reg_lambda = tf.placeholder(tf.float32, [], name='lambda')
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
        
        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
            # self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            # self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            # self.emb_mtx = self.emb_mtx * self.emb_mtx_mask
            
            # user interaction set and co-interaction set
            self.user_1hop = tf.nn.embedding_lookup(self.emb_mtx, self.user_1hop_ph)
            self.user_2hop = tf.nn.embedding_lookup(self.emb_mtx, self.user_2hop_ph)
            
            # item interaction set and co-interaction set
            self.item_1hop = tf.nn.embedding_lookup(self.emb_mtx, self.item_1hop_ph)
            self.item_2hop = tf.nn.embedding_lookup(self.emb_mtx, self.item_2hop_ph)
            
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
        pred, label, loss, auxloss = sess.run([self.y_pred, self.label_ph, self.loss, self.auxloss], feed_dict = {
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
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss, auxloss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))
    
    def co_attention(self, key, value, query):
        # key/value: [B, T, K, D], query: [B, T, D]
        key_shape = key.get_shape().as_list()
        query = tf.expand_dims(query, 2) # [B, T, 1, D]
        query = tf.tile(query, [1, 1, key_shape[2], 1]) #[B, T, K, D]
        query_key_concat = tf.concat([query, key], axis = 3)
        atten = tf.layers.dense(query_key_concat, 1, activation=None, use_bias=False) #[B, T, K, 1]
        atten = self.lrelu(atten)
        atten = tf.nn.softmax(atten, dim=2) #[B, T, K, 1]
        res = tf.reduce_sum(atten * value, axis=2)
        return res
    
    def lrelu(self, x, alpha=0.2):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    
    def interactive_attention(self, key, value, query):
        attention_inp = tf.concat([key, query], axis=-1)
        attention = tf.layers.dense(attention_inp, 10, activation=tf.nn.tanh)
        attention = tf.layers.dense(attention, 1, activation=tf.nn.tanh)
        attention = tf.nn.softmax(attention, dim=1)

        wei_sum = tf.reduce_sum(value * value, axis=1)
        return wei_sum, attention


class SCORE(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice):
        super(SCORE, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
        # HOP:0
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])
        user_side = self.target_user_t
        item_side = self.target_item_t

        # HOP:1
        user_1hop_co_attention = self.co_attention(self.user_1hop, self.user_1hop, item_side)
        item_1hop_co_attention = self.co_attention(self.item_1hop, self.item_1hop, user_side)
        user_side += user_1hop_co_attention
        item_side += item_1hop_co_attention
        
        # HOP:2
        user_2hop_co_attention = self.co_attention(self.user_2hop, self.user_2hop, item_side)
        item_2hop_co_attention = self.co_attention(self.item_2hop, self.item_2hop, user_side)
        user_side += user_2hop_co_attention
        item_side += item_2hop_co_attention
        
        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        user_side_final_state, self.user_side_atten = self.interactive_attention(item_side_rep_t, user_side_rep_t, user_side_rep_t)
        item_side_final_state, self.item_side_atten = self.interactive_attention(user_side_rep_t, item_side_rep_t, item_side_rep_t)
        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_user, self.target_item], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.auxloss = self.loss
        self.build_train_step()

class No_Att(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice):
        super(No_Att, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
        # HOP:0
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])
        user_side = self.target_user_t
        item_side = self.target_item_t

        # HOP:1
        user_1hop_co_attention = self.co_attention(self.user_1hop, self.user_1hop, item_side)
        item_1hop_co_attention = self.co_attention(self.item_1hop, self.item_1hop, user_side)
        user_side += user_1hop_co_attention
        item_side += item_1hop_co_attention
        
        # HOP:2
        user_2hop_co_attention = self.co_attention(self.user_2hop, self.user_2hop, item_side)
        item_2hop_co_attention = self.co_attention(self.item_2hop, self.item_2hop, user_side)
        user_side += user_2hop_co_attention
        item_side += item_2hop_co_attention
        
        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_user, self.target_item], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.auxloss = self.loss
        self.build_train_step()

class SCORE_1HOP(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice):
        super(SCORE_1HOP, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
        # HOP:0
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])
        user_side = self.target_user_t
        item_side = self.target_item_t

        # HOP:1
        user_1hop_co_attention = self.co_attention(self.user_1hop, self.user_1hop, item_side)
        item_1hop_co_attention = self.co_attention(self.item_1hop, self.item_1hop, user_side)
        user_side += user_1hop_co_attention
        item_side += item_1hop_co_attention
        
        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        user_side_final_state, self.user_side_atten = self.link_attention(item_side_rep_t, user_side_rep_t, user_side_rep_t)
        item_side_final_state, self.item_side_atten = self.link_attention(user_side_rep_t, item_side_rep_t, item_side_rep_t)
        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_user, self.target_item], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.auxloss = self.loss
        self.build_train_step()

class GAT(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice):
        super(GAT, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice)
        # HOP:0
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])
        user_side = self.target_user_t
        item_side = self.target_item_t

        # HOP:1
        user_1hop_gat = self.gat(self.user_1hop, self.user_1hop, user_side)
        item_1hop_gat = self.gat(self.item_1hop, self.item_1hop, item_side)
        user_side += user_1hop_gat
        item_side += item_1hop_gat
        
        # HOP:2
        user_2hop_gat = self.gat(self.user_2hop, self.user_2hop, user_side)
        item_2hop_gat = self.gat(self.item_2hop, self.item_2hop, item_side)
        user_side += user_2hop_gat
        item_side += item_2hop_gat
        
        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        user_side_final_state, self.user_side_atten = self.link_attention(item_side_rep_t, user_side_rep_t, user_side_rep_t)
        item_side_final_state, self.item_side_atten = self.link_attention(user_side_rep_t, item_side_rep_t, item_side_rep_t)
        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_user, self.target_item], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.auxloss = self.loss
        self.build_train_step()

    def lrelu(self, x, alpha=0.2):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
        
    def gat(self, key, value, query):
        # key/value: [B, T, K, D], query: [B, T, D]
        key_shape = key.get_shape().as_list()
        query = tf.expand_dims(query, 2) # [B, T, 1, D]
        query = tf.tile(query, [1, 1, key_shape[2], 1]) #[B, T, K, D]
        query_key_concat = tf.concat([query, key], axis = 3)
        atten = tf.layers.dense(query_key_concat, 1, activation=None, use_bias=False) #[B, T, K, 1]
        atten = self.lrelu(atten)
        atten = tf.nn.softmax(atten, dim=2) #[B, T, K, 1]
        res = tf.reduce_sum(atten * value, axis=2)
        return res

    
    def link_attention(self, key, value, query):
        attention_inp = tf.concat([key, query], axis=-1)
        attention = tf.layers.dense(attention_inp, 10, activation=tf.nn.tanh)
        attention = tf.layers.dense(attention, 1, activation=tf.nn.tanh)
        attention = tf.nn.softmax(attention, dim=1)

        wei_sum = tf.reduce_sum(value * value, axis=1)
        return wei_sum, attention