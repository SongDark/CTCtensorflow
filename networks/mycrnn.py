from universal import *

class mycrnn():
    def get_conv_result_seqlen(self, lens):
        for i in range(len(lens)):
            # conv1, maxpool1
            lens[i] = int((lens[i]-4)/2)
            # conv2, maxpool2
            lens[i] = int((lens[i]-4)/2)
            # conv3, maxpool3
            lens[i] = int((lens[i]-8)/2)
        return lens

    def __init__(self, params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('input'):
                # size is [batch_size, max_timestep, num_features, num_channels]
                # self.inputs = tf.placeholder(tf.float32, [None, None, num_features, num_channels])
                self.inputs = tf.placeholder(tf.float32, [None, None, params.num_features, params.expand_channel])

                # SparseTensor required by ctc_loss.
                self.targets = tf.sparse_placeholder(tf.int32)

                # 1d array of convolutional result length, size is [batch_size,]        
                self.seq_len = tf.placeholder(tf.int32, [None])

                # keep_prob for dropout layer
                self.keep_prob = tf.placeholder("float")

            with tf.name_scope('conv2d_maxpooling_1'):
                self.w_conv1 = random_variable_init(shape=[5,2,params.expand_channel,32],stddev=0.3)
                self.b_conv1 = random_variable_init(shape=[32],stddev=0.3)
                self.h_conv1 = tf.nn.relu(conv2d(self.inputs, self.w_conv1, strides=[1,1,1,1]) + self.b_conv1)
                self.h_pool1 = max_pooling(self.h_conv1, ksize=[1,2,1,1], strides=[1,2,1,1])
                print self.inputs.shape
                print self.h_conv1.shape
                print self.h_pool1.shape
            
            with tf.name_scope('conv2d_maxpooling_2'):
                self.w_conv2 = random_variable_init(shape=[5,2,32,32],stddev=0.3)
                self.b_conv2 = random_variable_init(shape=[32],stddev=0.3)
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, strides=[1,1,1,1]) + self.b_conv2)
                self.h_pool2 = max_pooling(self.h_conv2, ksize=[1,2,1,1], strides=[1,2,1,1])
                print self.h_conv2.shape
                print self.h_pool2.shape
            
            with tf.name_scope('conv2d_maxpooling_3'):
                self.w_conv3 = random_variable_init(shape=[5,2,32,32],stddev=0.3)
                self.b_conv3 = random_variable_init(shape=[32],stddev=0.3)
                self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.w_conv3, strides=[1,1,1,1]) + self.b_conv3)
                self.w_conv33 = random_variable_init(shape=[5,2,32,32],stddev=0.3)
                self.b_conv33 = random_variable_init(shape=[32],stddev=0.3)
                self.h_conv3 = tf.nn.relu(conv2d(self.h_conv3, self.w_conv33, strides=[1,1,1,1]) + self.b_conv33)
                self.h_pool3 = max_pooling(self.h_conv3, ksize=[1,2,1,1], strides=[1,2,1,1])
                print self.h_conv3.shape
                print self.h_pool3.shape
            # self.h_pool3 = self.h_pool2

            h_pool3_shape = tf.shape(self.h_pool3)
            batch_s, conv_result_length, conv_result_widith, num_feature_maps = h_pool3_shape[0], h_pool3_shape[1], int(self.h_pool3.shape[-2]), int(self.h_pool3.shape[-1])

            with tf.name_scope('reshape_1'):
                self.reshape_1 = tf.reshape(self.h_pool3, (batch_s, conv_result_length, conv_result_widith * num_feature_maps))
            
            with tf.name_scope('dense_1'):
                self.dense_1_input = tf.reshape(self.reshape_1, (-1, conv_result_widith * num_feature_maps))
                self.w_dense1 = random_variable_init(shape=[conv_result_widith * num_feature_maps, 32])
                self.b_dense1 = random_variable_init(shape=[32])
                self.dense_1 = tf.nn.relu(tf.matmul(self.dense_1_input, self.w_dense1) + self.b_dense1)
                self.dense_1 = tf.reshape(self.dense_1, (batch_s, conv_result_length, 32))

            with tf.name_scope('Bi_GRU_1'):
                # multi-layer bi-lstm, would raise Error if not distinguish variable_scope !
                with tf.variable_scope('Bi_GRU_1'): 
                    self.GRU_fw_cell_1 = tf.contrib.rnn.GRUCell(64)
                    self.GRU_bw_cell_1 = tf.contrib.rnn.GRUCell(64)
                    (self.Bi_GRU_fw_1, self.Bi_GRU_bw_1), (state_fw_1, state_bw_1) = tf.nn.bidirectional_dynamic_rnn(self.GRU_fw_cell_1, self.GRU_bw_cell_1, self.dense_1, self.seq_len, dtype=tf.float32)
                    self.Bi_GRU_1 = self.Bi_GRU_fw_1 + self.Bi_GRU_bw_1

            with tf.name_scope('Bi_GRU_2'):
                with tf.variable_scope('Bi_GRU_2'): 
                    self.GRU_fw_cell_2 = tf.contrib.rnn.GRUCell(64)
                    self.GRU_bw_cell_2 = tf.contrib.rnn.GRUCell(64)
                    (self.Bi_GRU_fw_2, self.Bi_GRU_bw_2), (state_fw_2, state_bw_2) = tf.nn.bidirectional_dynamic_rnn(self.GRU_fw_cell_2, self.GRU_bw_cell_2, self.Bi_GRU_1, self.seq_len, dtype=tf.float32)
                    self.Bi_GRU_2 = tf.concat([self.Bi_GRU_fw_2, self.Bi_GRU_bw_2], axis=2)
            
            with tf.name_scope('dropout'):
                # dropout by prob 0.5
                self.drop_out = tf.nn.dropout(self.Bi_GRU_2, keep_prob=self.keep_prob)
            
            with tf.name_scope('dense_2'):
                self.dense_2_input = tf.reshape(self.drop_out, (-1, 128))
                self.w_dense2 = random_variable_init([128, params.num_classes],stddev=0.3)
                self.b_dense2 = random_variable_init([params.num_classes],stddev=0.3)
                # fully-connected: [batch_size, conv_result_length, num_classes]
                self.dense_2 = tf.matmul(self.dense_2_input, self.w_dense2) + self.b_dense2
                # self.dense_2 = tf.nn.softmax(self.dense_2)
                self.dense_2 = tf.reshape(self.dense_2, (batch_s, conv_result_length, params.num_classes))
                # time major
                self.logits = tf.transpose(self.dense_2, (1,0,2))
            
            with tf.name_scope('ctc_loss'):
                self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len)
                self.cost = tf.reduce_mean(self.loss)
            
            with tf.name_scope('train'):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)                

            with tf.name_scope('error_rate'):
                self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len)
                # label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))