#import libraies
print('loading_libraries..')
import tensorflow as tf
import random 
import numpy as np

#hyperparameters

epoch          = 10
data_set_lenth = 100
batch_size     =5
iteration      =int(data_set_len//batch_size)

#data_loading
print('loading_data_files')

pos=np.load('positive_reviews_imbd.npy')
neg=np.load('negative_reviews_imbd.npy')

data_x  =np.load('word_embedding_lstm.npy')
words_y =np.load('words_list_lstm.npy')

mixed_data=[]
for i in pos:
    mixed_data.append((1,i))
for j in neg[:100]:
    mixed_data.append((0,j))

random.shuffle(mixed_data)

def padding_data(data_):
    input_x_data = []
    output_y_data = []
    max_value = max([len(j) for i, j in data_])

    final_data = [(j, i + [0] * (max_value - len(i))) 
                  if len(i) < max_value else (j, i) 
                  for j, i in data_]
    
    for i, j in final_data:
        input_x_data.append(j)
        output_y_data.append(i)

    return {'input': input_x_data, 'output': output_y_data}


class Lstm_sentiment_network(object):
    
    
    def __init__(self,vocab_size,word_embedding_dim,num_uni,labels):
        
        tf.reset_default_graph()
        
        sentence  = tf.placeholder(name='input_sentence',shape=[None,None],dtype=tf.int32)
        sentiment_= tf.placeholder(name='sentiment',shape=[None],dtype=tf.int32)
        
        self.placeholders = {'sentence':sentence,'sentiment':sentiment_}
        
        word_embedding = tf.get_variable(name='word_embedding_',
                                         shape=[vocab_size,word_embedding_dim],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        embedding_lookup = tf.nn.embedding_lookup(word_embedding,sentence)
        
        sequence_leng = tf.count_nonzero(sentence,axis=-1)
        
        with tf.variable_scope('forward'):
            fr_cell = tf.contrib.rnn.LSTMCell(num_units=num_uni)
            dropout_fr = tf.contrib.rnn.DropoutWrapper(fr_cell)
            
        with tf.variable_scope('backward'):
            bw_cell = tf.contrib.rnn.LSTMCell(num_units=num_uni)
            dropout_bw = tf.contrib.rnn.DropoutWrapper(bw_cell)
            
        with tf.variable_scope('encoder') as scope:
            model,last_state = tf.nn.bidirectional_dynamic_rnn(dropout_fr,
                                                               dropout_bw,
                                                               inputs=embedding_lookup,
                                                               sequence_length=sequence_leng,
                                                               dtype=tf.float32)
            
        concat_output = tf.concat([last_state[0].c,last_state[1].c],axis=-1)
        
        fc_layer = tf.get_variable(name='fully_connected',
                                   shape=[2*num_uni,labels],
                                   dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        bias    = tf.get_variable(name='bias',
                                   shape=[labels],
                                   dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        projection = tf.add(tf.matmul(concat_output,fc_layer),bias)
        
        #prediction
        probability = tf.nn.softmax(projection)
        prediction  = tf.argmax(probability,axis=-1)
        
        
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=projection,labels=sentiment_)
        
        cost = tf.reduce_mean(cross_entropy)
        
        #accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prediction,tf.int32),sentiment_),tf.float32))
        
        
        self.output = {'loss':cost,'accuracy':accuracy,'logits': projection,'check1':embedding_lookup,'check2':model,'check3':last_state,'projection':projection,'con':concat_output}
        self.train = tf.train.AdamOptimizer().minimize(cost)
        
        
def model_execute(model):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        model_out,train = sess.run([model.output,model.train],feed_dict={model.placeholders['sentence']:np.random.randint(0,10,[12,13]),model.placeholders['sentiment']:np.random.randint(0,2,[12,])})
        
        
        print(model_out['loss'],model_out['accuracy'],model_out['logits'].shape,model_out['check1'].shape,model_out['check2'][0].shape,model_out['check3'][0].c.shape,model_out['projection'].shape,model_out['con'].shape)
        
        
if __name__=='__main__':
    
    model_out_ = Lstm_sentiment_network(121,102,13,2)
    model_execute(model_out_)
    
    



