from universal import *

num_hidden = 128

class mylstm():
    def get_conv_result_seqlen(self, lens):
        return lens

    def __init__(self, params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('input'):
                # size is [batch_size, max_timestep, num_features]
                self.inputs = tf.placeholder(tf.float32, [None, None, params.num_features, params.expand_channel])
                # SparseTensor required by ctc_loss.
                self.targets = tf.sparse_placeholder(tf.int32)
                # 1d array of labeling length, size is [batch_size,]        
                self.seq_len = tf.placeholder(tf.int32, [None])

                self.keep_prob = tf.placeholder("float")
            
            shape = tf.shape(self.inputs)
            batch_s, max_timestep = shape[0], shape[1]

            print self.inputs.shape
            with tf.name_scope('reshape'):
                self.inputs_reshape = tf.reshape(self.inputs, [batch_s, max_timestep, params.num_features * params.expand_channel])
            print self.inputs_reshape.shape
        
            with tf.name_scope('rnn_layer'):
                with tf.variable_scope('lstm'):
                    # Stacking the RNN cells                        
                    self.stacked_rnn = tf.contrib.rnn.MultiRNNCell([self.my_rnn_cell() for _ in range(2)], state_is_tuple=True)
                    # The 2nd part is the last state, not used.
                    self.outputs, _ = tf.nn.dynamic_rnn(self.stacked_rnn, self.inputs_reshape, self.seq_len, dtype=tf.float32)
                    self.outputs = tf.nn.dropout(self.outputs, keep_prob=self.keep_prob)

            # reshape to apply the same weights over all timesteps
            self.outputs = tf.reshape(self.outputs, [-1, num_hidden]) # [batch_size * max_timestep, num_hidden]
            
            with tf.name_scope('output_layer'):
                with tf.name_scope('w_dense2'):
                    self.w_dense2 = random_variable_init(shape=[num_hidden, params.num_classes], stddev=0.3)
                    # self.variable_summaries(self.w_dense2)
                with tf.name_scope('b_dense2'):
                    self.b_dense2 = random_variable_init([params.num_classes], 0.5)
                    # self.variable_summaries(self.b_dense2)
                with tf.name_scope('Wx_plus_b_2'):
                    self.logits = tf.matmul(self.outputs, self.w_dense2) + self.b_dense2 # [batch_size * max_timestep, num_classes]
                    # reshape back to the original shape
                    self.logits = tf.reshape(self.logits, [batch_s, -1, params.num_classes]) # [batch_size, max_timestep, num_hidden]

                    # softmax may cause inf loss
                    # self.logits = tf.nn.log_softmax(self.logits, dim=-1)

                    # time major
                    self.logits = tf.transpose(self.logits,(1,0,2)) # [max_timestep, batch_size, num_hidden]

                    # tf.summary.histogram('logits', self.logits)

            with tf.name_scope('ctc_loss'):
                self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len)
                with tf.name_scope('total_cost'):
                    self.cost = tf.reduce_mean(self.loss)
                    # tf.summary.scalar('ctc_cost', self.cost)
            
            with tf.name_scope('train'):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

            with tf.name_scope('error_rate'):
                self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len)

                # label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))
            
            # rnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lstm')
            # output_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer')
            # self.variable_summaries(rnn_variables[1])

    def my_rnn_cell(self):
        # Define RNN cell
                # can also be:
                #   tf.nn.rnn_cell.RNNCell
                #   tf.nn.rnn_cell.GRUCell
        # cell parameters (no peepholes):
        #       kernel: [num_features, 4 * num_hidden] + [num_hidden, 4 * num_hidden] (i,j,f,o)
        #       bias: [4 * num_hidden]
        # (with peepholes):
        # w_f, w_i, w_o
        return tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True, forget_bias=0.5)

# lstm = mylstm()
