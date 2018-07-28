
# coding: utf-8

# In[ ]:



import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
# from attention import attention


class LSTM_Network(object):
    """
      Lstm Network
    """
    def __init__(self, vocab_size_, dropout_value_, embedding_dim_, forget_bias_, rnn_num_units):
        """
        Multi layer Bi-directional LSTM
        In a single layer RNN, the output is produced by passing it through a single hidden
        state which fails to capture hierarchical (think temporal) structure of a sequence.
        With a multi-layered RNN, such structure is captured which results in better performance.
        :param vocab_size_:
              vocabulary size
        :param dropout_value_:
              dropout for neurons
        :param embedding_dim_:
              word_embedding dimension
        :param forget_bias_:
              forget_bias for LSTM cell gates {default : 1}
              Including a bias of 1 to the forget gate of every LSTM cell is also shown to improve performance.
        :param rnn_num_units:
              # of units Lstm cell
        :return
             loss
             prediction
             probability
             logits
             accuracy
        :raise:
            if input shape is different from placeholder shape:
                ValueError: Cannot feed value of shape
            if Any layer output shape is not compatible for next layer input shape :
                { ex : output shape of rnn to input shape of fully connected layer }
                ValueError: Dimensions must be equal
            if not reset the graph:
               ValueError: Attempt to have a second RNNCell use the
               weights of a variable scope that already has weights
        """
        tf.reset_default_graph()


        # placeholders
        sentence_input = tf.placeholder(
            name='input',
            shape=[None, None],
            dtype=tf.int32)

        label_s = tf.placeholder(
            name='output',
            shape=[None, ],
            dtype=tf.int32

        )

        dropout_mode = tf.placeholder(
            name='mode',
            shape=(),
            dtype=tf.int32
        )

        self.placeholder = {

            'input': sentence_input,
            'label_s': label_s,
            'dropout_mode': dropout_mode

        }

        sequence_len = tf.count_nonzero(sentence_input, axis=-1)
        # sequence length for rnn unfolding

        dropout = tf.cond(
            tf.equal(dropout_mode, 0),  # If
            lambda: dropout_value_,  # True
            lambda: 0.  # False
        )

        # word_embedding and embedding lookup
        word_embedding_ = tf.get_variable(name='embeddings_',
                                          shape=[vocab_size_, embedding_dim_],
                                          dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(-0.01, 0.01))

        embedding_lookup = tf.nn.embedding_lookup(word_embedding_, sentence_input)

        with tf.variable_scope('encoder'):
            # forward cell of Bi-directional lstm network



            def fr_cell():
                fr_cell_lstm = rnn.LSTMCell(num_units=rnn_num_units, forget_bias=forget_bias_)

                return rnn.DropoutWrapper(cell=fr_cell_lstm, output_keep_prob=1. - dropout, dtype=tf.float32)
                # dropout layer for forward cell

            # Forward RNNCells as its inputs and wraps them into a single cell

            fr_cell_m = rnn.MultiRNNCell([fr_cell() for _ in range(1)], state_is_tuple=True)
        # fr_initial_cell = fr_cell_m.zero_state(batch_size=batch_size,dtype=tf.float32)



        with tf.variable_scope('encoder'):
            # backward cell for Bi-directional lstm network



            def bw_cell():
                bw_cell_lstm = rnn.LSTMCell(num_units=rnn_num_units, forget_bias=forget_bias_)

                return rnn.DropoutWrapper(cell=bw_cell_lstm, output_keep_prob=1. - dropout, dtype=tf.float32)
                # droput layer for backward cell

            # Backward RNNCells as its inputs and wraps them into a single cell

            bw_cell_m = rnn.MultiRNNCell([bw_cell() for _ in range(1)], state_is_tuple=True)


            # bw_initial_cell = bw_cell_m.zero_state(batch_size=batch_size,dtype=tf.float32)


            # return of Bi-directional  lstm network  :

            # A tuple (outputs, output_states) where:

            # outputs       :        A tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor #batch_size, max_time, cell_fw.output_size]
            # output_states :        A tuple ( state.cT , state.hT ) containing the shape [Batch_size,num_inputs] and has the final cell state cT and output state hT of each batch sequence.

        # Bi-directional lstm network
        with tf.variable_scope('encoder') as scope:
            model, (state_c, state_h) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fr_cell_m,  # forward cell
                cell_bw=bw_cell_m,  # backward cell
                inputs=embedding_lookup,  # 3 dim embedding input for rnn
                sequence_length=sequence_len,  # sequence len == batch_size

                # initial_state_fw=fr_initial_cell,

                # initial_state_bw=bw_initial_cell,
                dtype=tf.float32

            )

        transpose = tf.concat(model, 2)
        
        #Attention_layer 
        
        x_attention = tf.reshape(transpose,[-1,rnn_num_units*2])
        attention_size=tf.get_variable(name='attention',shape=[rnn_num_units*2,1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        bias_ = tf.get_variable(name='bias_',shape=[1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        linear_projection = tf.add(tf.matmul(x_attention,attention_size),bias_)
#         print(sentence_input.shape[0])
        reshape_ = tf.reshape(linear_projection,[tf.shape(sentence_input)[0],tf.shape(sentence_input)[1],-1])
        attention_output=tf.nn.softmax(reshape_,dim=1)
        
        atten_visualize=tf.reshape(attention_output,[tf.shape(sentence_input)[0],tf.shape(sentence_input)[1]],name='plot_dis')
        
        multi = tf.multiply(attention_output,transpose)
        

        atten_out_s = tf.reduce_sum(multi,1)

#         attention_visualize = tf.reshape(atten_out,[tf.shape(sentence_input)[0],tf.shape(sentence_input)[1]])
        
        
        
        
        
        


        

        
        
        
        

#         state_output = tf.concat([state_c[0].c, state_h[0].c], axis=-1)

        # Attention Mechanism

        # Attention_output,alphas= attention(transpose ,30, return_alphas=True,time_major=False)


#         # will return [batch_size, output_state_size]
        weights = tf.get_variable(name='weights',
                                  shape=[2*rnn_num_units, 2],
                                  dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-0.01, 0.01))

        bias = tf.get_variable(name='bias',
                               shape=[2],
                               dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-0.01, 0.01))

        logits = tf.add(tf.matmul(atten_out_s, weights),
                        bias, name='network_output')

# #         # self.check_shapes={'embedding':embedding_lookup,'model':model,'tr':transpose,'atten':attention_output,'al':alphas,'outa':mat_out}

        probability_distribution = tf.nn.softmax(logits, name='netout')

        prediction = tf.argmax(probability_distribution, axis=-1)

        # cross entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_s
        )
        # loss
        loss = tf.reduce_mean(ce)

        # accuracy calculation

        accuracy_calculation = tf.reduce_mean(tf.cast(
            tf.equal(
                tf.cast(prediction, tf.int32), label_s),
            tf.float32))

        self.output = {


            'loss': loss,
            'prob': probability_distribution,
            'pred': prediction,
            'logits': logits,
            'accuracy': accuracy_calculation

        }

        self.train = tf.train.AdamOptimizer().minimize(loss)


# uncomment below code for demo


# def checking_model(model):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         out_now=sess.run(model.output,feed_dict={model.placeholder['input']:np.random.randint(0,10,[16,10]),model.placeholder['label_s']:np.random.randint(0,2,[16,]),model.placeholder['dropout_mode']:0})
#         print(out_now['loss'].shape)
#         print(out_now['prob'].shape)
        


# if __name__=="__main__":

#     model=LSTM_Network(10,0.5,2,1.0,6)

#     checking_model(model)

