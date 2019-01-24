import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

'''
Slice Based Models: SCORE, RRN
'''
class SliceBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        # reset graph
        tf.reset_default_graph()

        # k items/users/ per time slice
        self.K = k
        self.neg_num = 4

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_1hop_ph = tf.placeholder(tf.int32, [None, max_len, self.K, item_fnum], name='user_1hop_ph')
            self.user_2hop_ph = tf.placeholder(tf.int32, [None, max_len, self.K, user_fnum], name='user_2hop_ph')
            self.item_1hop_ph = tf.placeholder(tf.int32, [None, max_len, self.K, user_fnum], name='item_1hop_ph')
            self.item_2hop_ph = tf.placeholder(tf.int32, [None, max_len, self.K, item_fnum], name='item_2hop_ph')

            self.user_1hop_len_ph = tf.placeholder(tf.int32, [None,], name='user_1hop_len_ph')
            self.user_2hop_len_ph = tf.placeholder(tf.int32, [None,], name='user_2hop_len_ph')
            self.item_1hop_len_ph = tf.placeholder(tf.int32, [None,], name='item_1hop_len_ph')
            self.item_2hop_len_ph = tf.placeholder(tf.int32, [None,], name='item_2hop_len_ph')

            self.user_1hop_border_ph = tf.placeholder(tf.int32, [None, max_len, self.K], name='user_1hop_border_ph')
            self.user_2hop_border_ph = tf.placeholder(tf.int32, [None, max_len, self.K], name='user_2hop_border_ph')
            self.item_1hop_border_ph = tf.placeholder(tf.int32, [None, max_len, self.K], name='item_1hop_border_ph')
            self.item_2hop_border_ph = tf.placeholder(tf.int32, [None, max_len, self.K], name='item_2hop_border_ph')

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
            self.emb_mtx_mask = tf.concat([self.emb_mtx_mask, tf.constant(value=0., shape=[1, eb_dim])], axis=0)
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
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.tanh, name='fc2')
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
        self.pred_reshape = tf.reshape(self.y_pred, [-1, self.neg_num + 1])
        self.pred_pos = tf.tile(tf.expand_dims(self.pred_reshape[:, 0], 1), [1, self.neg_num])
        self.pred_neg = self.pred_reshape[:, 1:]
        self.loss = tf.reduce_mean(tf.log(tf.nn.sigmoid(self.pred_pos - self.pred_neg)))
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.label_ph : batch_data[0],
                self.target_user_ph : batch_data[1],
                self.target_item_ph : batch_data[2],
                self.user_1hop_ph : batch_data[3],
                self.user_2hop_ph : batch_data[4],
                self.item_1hop_ph : batch_data[5],
                self.item_2hop_ph : batch_data[6],
                self.user_1hop_len_ph : batch_data[7],
                self.user_2hop_len_ph : batch_data[8],
                self.item_1hop_len_ph : batch_data[9],
                self.item_2hop_len_ph : batch_data[10],
                self.user_1hop_border_ph : batch_data[11],
                self.user_2hop_border_ph : batch_data[12],
                self.item_1hop_border_ph : batch_data[13],
                self.item_2hop_border_ph : batch_data[14],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data):
        pred, label = sess.run([self.y_pred, self.label_ph], feed_dict = {
                self.label_ph : batch_data[0],
                self.target_user_ph : batch_data[1],
                self.target_item_ph : batch_data[2],
                self.user_1hop_ph : batch_data[3],
                self.user_2hop_ph : batch_data[4],
                self.item_1hop_ph : batch_data[5],
                self.item_2hop_ph : batch_data[6],
                self.user_1hop_len_ph : batch_data[7],
                self.user_2hop_len_ph : batch_data[8],
                self.item_1hop_len_ph : batch_data[9],
                self.item_2hop_len_ph : batch_data[10],
                self.user_1hop_border_ph : batch_data[11],
                self.user_2hop_border_ph : batch_data[12],
                self.item_1hop_border_ph : batch_data[13],
                self.item_2hop_border_ph : batch_data[14],
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist()
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))
    
    def co_attention(self, seq1, seq2, border1, border2):
        seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1], activation=tf.nn.relu)
        seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1])
        seq2 = tf.layers.dense(seq2, seq2.get_shape().as_list()[-1], activation=tf.nn.relu)
        # seq1 = tf.layers.dense(seq1, seq1.get_shape().as_list()[-1])
        product = tf.matmul(seq1, tf.transpose(seq2, [0, 1, 3, 2]))

        # get 2 masks
        mask1 = tf.transpose(tf.sequence_mask(border1, self.K, tf.float32), [0, 1, 3, 2])
        mask2 = tf.sequence_mask(border2, self.K, tf.float32)

        product = product * mask1 * mask2

        mask1 = (1 - mask1) * -1e10
        mask2 = (1 - mask2) * -1e10

        product = product + mask1 + mask2

        # seq1_weights = tf.expand_dims(tf.ones_like(tf.reduce_max(product, axis=3)), axis=3)
        # seq2_weights = tf.expand_dims(tf.ones_like(tf.reduce_max(product, axis=2)), axis=3)
        seq1_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=3)), axis=3)
        seq2_weights = tf.expand_dims(tf.nn.softmax(tf.reduce_max(product, axis=2)), axis=3)

        seq1_result = tf.reduce_sum(seq1 * seq1_weights, axis=2) #[B, T, D]
        seq2_result = tf.reduce_sum(seq2 * seq2_weights, axis=2)

        return seq1_result, seq2_result

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

class SCORE(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(SCORE, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)
        
        # co-attention
        with tf.name_scope('co-attention'):
            user_1hop_seq, item_2hop_seq = self.co_attention(self.user_1hop, self.item_2hop, self.user_1hop_border_ph, self.item_2hop_border_ph)
            user_2hop_seq, item_1hop_seq = self.co_attention(self.user_2hop, self.item_1hop, self.user_2hop_border_ph, self.item_1hop_border_ph)

            user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
            user_side = tf.layers.dense(user_side, user_side.get_shape().as_list()[-1], activation=tf.nn.relu)
            item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)
            item_side = tf.layers.dense(item_side, item_side.get_shape().as_list()[-1], activation=tf.nn.relu)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1) #

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()


class SCORE_Att(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(SCORE_Att, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)
        self.mask = tf.sequence_mask(self.user_1hop_len_ph, max_len, dtype=tf.float32) # [B, T]
        self.mask = tf.expand_dims(self.mask, -1) # [B, T, 1]

        # co-attention
        with tf.name_scope('co-attention'):
            user_1hop_seq, item_2hop_seq = self.co_attention(self.user_1hop, self.item_2hop, self.user_1hop_border_ph, self.item_2hop_border_ph)
            user_2hop_seq, item_1hop_seq = self.co_attention(self.user_2hop, self.item_1hop, self.user_2hop_border_ph, self.item_1hop_border_ph)

            user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
            item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            user_out, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            item_out, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')
        with tf.name_scope('att'):
            query = tf.concat([self.target_item, self.target_user], axis=1)
            user_side_final_state, _, _ = self.attention(user_out, user_out, query, self.mask)
            item_side_final_state, _, _ = self.attention(item_out, item_out, query, self.mask)
        
        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1) #, self.target_user

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class SCORE_B(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(SCORE_B, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)

        # co-attention
        with tf.name_scope('co-attention'):
            user_1hop_seq, item_2hop_seq = self.co_attention(self.user_1hop, self.item_2hop, self.user_1hop_border_ph, self.item_2hop_border_ph)
            user_2hop_seq, item_1hop_seq = self.co_attention(self.user_2hop, self.item_1hop, self.user_2hop_border_ph, self.item_1hop_border_ph)

            user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
            item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            user_side_middle_out, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            item_side_middle_out, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')

        # Probability of Graph
        user_side_graph_ps = tf.reshape(tf.nn.sigmoid(tf.layers.dense(user_side_middle_out, 1, name='g_fc1')), [-1, max_len])
        item_side_graph_ps = tf.reshape(tf.nn.sigmoid(tf.layers.dense(item_side_middle_out, 1, name='g_fc2')), [-1, max_len])

        user_side_graph_p_cum = tf.cumprod(user_side_graph_ps, axis=1)
        item_side_graph_p_cum = tf.cumprod(item_side_graph_ps, axis=1)
        
        idx = tf.stack([tf.reshape(tf.range(tf.shape(self.user_1hop_len_ph)[0]), (-1,1)), tf.reshape(self.user_1hop_len_ph - 1, [-1, 1])], axis=-1)
        self.user_side_graph_p = tf.reshape(tf.gather_nd(user_side_graph_p_cum, idx), [-1,])
        self.item_side_graph_p = tf.reshape(tf.gather_nd(item_side_graph_p_cum, idx), [-1,])

        # fc layer
        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1) #, self.target_user
        self.build_fc_net(inp, self.user_side_graph_p, self.item_side_graph_p)
        
        # loss
        self.build_bayesianloss()
    
    def build_bayesianloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_fc_net(self, inp, p_user, p_item):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.tanh, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1,]) * p_user * p_item

class RRN(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(RRN, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)

        # sum pooling
        user_1hop_seq = tf.reduce_sum(self.user_1hop, axis=2)
        item_1hop_seq = tf.reduce_sum(self.item_1hop, axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_1hop_seq, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            _, item_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_1hop_seq, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')
        
        inp = tf.concat([item_seq_final_state, user_seq_final_state, self.target_item, self.target_user], axis=1) #

        # fc layer
        self.build_fc_net(inp)
        self.build_mseloss()


class RRN_WS(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(RRN_WS, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)
        self.max_len = max_len

        # weighted sum pooling
        user_1hop_seq = self.weighted_sum(self.user_1hop)
        item_1hop_seq = self.weighted_sum(self.item_1hop)
        
        
        # RNN
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_1hop_seq, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            _, item_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_1hop_seq, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')
        
        inp = tf.concat([item_seq_final_state, user_seq_final_state, self.target_item, self.target_user], axis=1) #, self.target_user

        # fc layer
        self.build_fc_net(inp)
        self.build_mseloss()
    
    def weighted_sum(self, seq):
        seq_weight = tf.layers.dense(seq, 1, activation=tf.nn.relu) #[B, T, K, 1]
        seq_weight = tf.reshape(seq_weight, [-1, self.max_len, self.K])
        seq_weight = tf.expand_dims(tf.nn.softmax(seq_weight), axis=3) #[B, T, K, 1]
        res = tf.reduce_sum(seq * seq_weight, axis=2)
        return res


class RRN_2HOP(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(RRN_2HOP, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)

        # sum pooling
        user_1hop_seq = tf.reduce_sum(self.user_1hop, axis=2)
        user_2hop_seq = tf.reduce_sum(self.user_2hop, axis=2)
        item_1hop_seq = tf.reduce_sum(self.item_1hop, axis=2)
        item_2hop_seq = tf.reduce_sum(self.item_2hop, axis=2)

        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            _, item_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')
        
        inp = tf.concat([item_seq_final_state, user_seq_final_state, self.target_item, self.target_user], axis=1) #, self.target_user

        # fc layer
        self.build_fc_net(inp)
        self.build_mseloss()

class GCMC(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum):
        super(GCMC, self).__init__(feature_size, eb_dim, hidden_size, max_len, k, user_fnum, item_fnum)

        user_1hop_li = tf.layers.dense(self.user_1hop, self.user_1hop.get_shape().as_list()[-1], activation=None, use_bias=False)
        item_1hop_li = tf.layers.dense(self.item_1hop, self.item_1hop.get_shape().as_list()[-1], activation=None, use_bias=False)

        # sum pooling
        user_1hop_seq_sum = tf.nn.relu(tf.reduce_sum(user_1hop_li, axis=2))
        item_1hop_seq_sum = tf.nn.relu(tf.reduce_sum(item_1hop_li, axis=2))

        user_1hop_seq = tf.layers.dense(user_1hop_seq_sum, user_1hop_seq_sum.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        item_1hop_seq = tf.layers.dense(item_1hop_seq_sum, item_1hop_seq_sum.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)

        # RNN
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_1hop_seq, 
                                                        sequence_length=self.user_1hop_len_ph, dtype=tf.float32, scope='gru1')
            _, item_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_1hop_seq, 
                                                        sequence_length=self.item_1hop_len_ph, dtype=tf.float32, scope='gru2')
        
        # inp = tf.concat([item_seq_final_state, user_seq_final_state, self.target_user, , self.target_item], axis=1) #, self.target_user

        # pred
        self.y_pred_pos = tf.exp(tf.reduce_sum(tf.layers.dense(item_seq_final_state, hidden_size, use_bias=False) * user_seq_final_state, axis=1))
        self.y_pred_neg = tf.exp(tf.reduce_sum(tf.layers.dense(item_seq_final_state, hidden_size, use_bias=False) * user_seq_final_state, axis=1))
        self.y_pred = self.y_pred_pos / (self.y_pred_pos + self.y_pred_neg)

        self.build_logloss()

'''
Point Based Models: GRU4Rec, (ARNN?)
'''
class PointBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, user_fnum, item_fnum):
        # reset graph
        tf.reset_default_graph()

        self.neg_num = 4

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_seq_ph = tf.placeholder(tf.int32, [None, max_len, item_fnum], name='user_seq_ph')
            self.item_seq_ph = tf.placeholder(tf.int32, [None, max_len, user_fnum], name='item_seq_ph')

            self.user_seq_len_ph = tf.placeholder(tf.int32, [None,], name='user_seq_len_ph')
            self.item_seq_len_ph = tf.placeholder(tf.int32, [None,], name='item_seq_len_ph')

            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob of dropout
            self.keep_prob = tf.placeholder(tf.float32, [])

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)

            self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_seq_ph)
            user_seq_shape = self.user_seq.get_shape().as_list()
            self.user_seq = tf.reshape(self.user_seq, [-1, user_seq_shape[1], user_seq_shape[2] * user_seq_shape[3]])

            self.item_seq = tf.nn.embedding_lookup(self.emb_mtx, self.item_seq_ph)
            item_seq_shape = self.item_seq.get_shape().as_list()
            self.item_seq = tf.reshape(self.item_seq, [-1, item_seq_shape[1], item_seq_shape[2] * item_seq_shape[3]])

            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
            target_item_shape = self.target_item.get_shape().as_list()
            self.target_item = tf.reshape(self.target_item, [-1, target_item_shape[1] * target_item_shape[2]])

            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            target_user_shape = self.target_user.get_shape().as_list()
            self.target_user = tf.reshape(self.target_user, [-1, target_user_shape[1] * target_user_shape[2]])


    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.tanh, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1,])
    
    def build_bprloss(self):
        self.pred_reshape = tf.reshape(self.y_pred, [-1, self.neg_num + 1])
        self.pred_pos = tf.tile(tf.expand_dims(self.pred_reshape[:, 0], 1), [1, self.neg_num])
        self.pred_neg = self.pred_reshape[:, 1:]
        self.loss = -tf.reduce_mean(tf.log(self.pred_pos - self.pred_neg))
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
    
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
    
    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.label_ph : batch_data[0],
                self.target_user_ph : batch_data[1],
                self.target_item_ph : batch_data[2],
                self.user_seq_ph : batch_data[3],
                self.user_seq_len_ph : batch_data[4],
                self.item_seq_ph : batch_data[5],
                self.item_seq_len_ph : batch_data[6],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss

    def eval(self, sess, batch_data):
        pred, label = sess.run([self.y_pred, self.label_ph], feed_dict = {
                self.label_ph : batch_data[0],
                self.target_user_ph : batch_data[1],
                self.target_item_ph : batch_data[2],
                self.user_seq_ph : batch_data[3],
                self.user_seq_len_ph : batch_data[4],
                self.item_seq_ph : batch_data[5],
                self.item_seq_len_ph : batch_data[6],
                self.keep_prob : 1.0
            })
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist()
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

class GRU4Rec(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, user_fnum, item_fnum):
        super(GRU4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_len, user_fnum, item_fnum)

        # RNN
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_len_ph, dtype=tf.float32, scope='gru1')
        
        inp = tf.concat([user_seq_final_state, self.target_item, self.target_user], axis=1) #, self.target_user

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class GRU4Rec_DUAL(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len, user_fnum, item_fnum):
        super(GRU4Rec_DUAL, self).__init__(feature_size, eb_dim, hidden_size, max_len, user_fnum, item_fnum)

        # RNN
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_len_ph, dtype=tf.float32, scope='gru1')
            _, item_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_seq, 
                                                        sequence_length=self.item_seq_len_ph, dtype=tf.float32, scope='gru2')
        
        inp = tf.concat([user_seq_final_state, item_seq_final_state, self.target_item, self.target_user], axis=1) #, self.target_user

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()


class SVDpp(object):
    def __init__(self, feature_size, eb_dim, max_len, user_fnum, item_fnum):
        # reset graph
        tf.reset_default_graph()

        # average score
        self.average = 0.2

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_seq_ph = tf.placeholder(tf.int32, [None, max_len, item_fnum], name='user_seq_ph')
            self.user_seq_len_ph = tf.placeholder(tf.int32, [None,], name='user_seq_len_ph')

            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim])#, initializer=tf.truncated_normal_initializer)

            self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_seq_ph)
            user_seq_shape = self.user_seq.get_shape().as_list()
            self.user_seq = tf.reshape(self.user_seq, [-1, user_seq_shape[1], user_seq_shape[2] * user_seq_shape[3]]) #[B, T, 2D]

            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
            target_item_shape = self.target_item.get_shape().as_list()
            self.target_item = tf.reshape(self.target_item, [-1, target_item_shape[1] * target_item_shape[2]]) #[B, 2D]
            
            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            target_user_shape = self.target_user.get_shape().as_list()
            self.target_user = tf.reshape(self.target_user, [-1, target_user_shape[1] * target_user_shape[2]]) #[B, D]
            
        # user and item bias
        with tf.name_scope('b'):
            self.item_user_bias = tf.get_variable('item_b', [feature_size, 1])
        
        # get pred score
        self.user_seq_mask = tf.expand_dims(tf.sequence_mask(self.user_seq_len_ph, max_len, dtype=tf.float32), 2)
        self.user_seq = self.user_seq * self.user_seq_mask
        self.neighbor = tf.reduce_sum(self.user_seq, axis=1)
        self.norm_neighbor_item = self.neighbor[:,:eb_dim] / tf.sqrt(tf.expand_dims(tf.norm(self.user_seq[:,:,:eb_dim], 1, (1, 2)), 1))
        self.norm_neighbor_cate = self.neighbor[:,eb_dim:] / tf.sqrt(tf.expand_dims(tf.norm(self.user_seq[:,:,eb_dim:], 1, (1, 2)), 1))

        self.latent_score = tf.reduce_sum((self.target_item[:,:eb_dim] + self.target_item[:, eb_dim:]) * (self.target_user + self.norm_neighbor_item + self.norm_neighbor_cate), 1) #[B,] #self.target_user + 
        self.user_bias = tf.reshape(tf.nn.embedding_lookup(self.item_user_bias, self.target_user_ph), [-1,])
        self.item_bias = tf.reshape(tf.nn.embedding_lookup(self.item_user_bias, self.target_item_ph[:,0]), [-1,])

        self.y_pred = tf.nn.sigmoid(self.average + self.user_bias + self.item_bias + self.latent_score)
        self.loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.label_ph : batch_data[0],
                self.target_user_ph : batch_data[1],
                self.target_item_ph : batch_data[2],
                self.user_seq_ph : batch_data[3],
                self.user_seq_len_ph : batch_data[4],
                self.lr : lr,
                self.reg_lambda : reg_lambda
            })
        return loss

    def eval(self, sess, batch_data):
        pred, label = sess.run([self.y_pred, self.label_ph], feed_dict = {
                self.label_ph : batch_data[0],
                self.target_user_ph : batch_data[1],
                self.target_item_ph : batch_data[2],
                self.user_seq_ph : batch_data[3],
                self.user_seq_len_ph : batch_data[4]
            })
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist()
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))