import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random

pos=np.load('positive_reviews_imbd.npy')
neg=np.load('negative_reviews_imbd.npy')

epoch=10
data_set_len=100
batch_size=5
iteration=int(data_set_len//batch_size)

mixed_data=[]
for i in pos:
    mixed_data.append((1,i))
for j in neg[:100]:
      mixed_data.append((0,j))

random.shuffle(mixed_data)
random.shuffle(mixed_data)


def padding_data(data_):
    input_x_data = []
    output_y_data = []
    max_value = max([len(j) for i, j in data_])

    final_data = [(j, i + [0] * (max_value - len(i))) if len(i) < max_value else (j, i) for j, i in data_]
    for i, j in final_data:
        input_x_data.append(j)
        output_y_data.append(i)

    return {'input': input_x_data, 'output': output_y_data}


labels=[1,0]


input_x=tf.placeholder(tf.int32,shape=[None,None])

output=tf.placeholder(tf.int32,shape=[None,])

data_x=np.load('word_embedding_lstm.npy')
words_y=np.load('words_list_lstm.npy')

word_embedding=tf.get_variable('W',shape=[400000,100],dtype=tf.float32,initializer=tf.constant_initializer(np.array(data_x)),trainable=False)
embedding_lookup=tf.nn.embedding_lookup(word_embedding,input_x)

sequ_len=tf.count_nonzero(input_x,axis=-1)

with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(num_units=250)
    model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=embedding_lookup,sequence_length=sequ_len,dtype=tf.float32)

model_output,(fs,fc)=model

concat_out=tf.concat((fs.c,fc.c),axis=-1)

fc_layer= tf.get_variable('weight',shape=[2*250,len(labels)],dtype=tf.float32,initializer=tf.random_normal_initializer(-0.01,0.01))
bias= tf.get_variable('bias',shape=[len(labels)],dtype=tf.float32,initializer=tf.random_normal_initializer(-0.01,0.01))

logi_ts=tf.matmul(concat_out,fc_layer)+bias

#normalization
prob=tf.nn.softmax(logi_ts)
pred=tf.argmax(prob,axis=-1)

#cross_entropy
ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logi_ts,labels=output)
loss=tf.reduce_mean(ce)


#accuracy
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred,tf.int32),output),tf.float32))

#train
train_x=tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        for j in range(iteration):
            slicing_data = mixed_data[j * batch_size:(j + 1) * batch_size]
            aa,bb,cc,dd,ee,ff,gg,hh,ggg=sess.run([model,concat_out,logi_ts,prob,pred,ce,loss,accuracy,train_x],feed_dict={input_x:padding_data(slicing_data)['input'],output:padding_data(slicing_data)['output']})

            print("eproch {} iteration {} loss {} accuracy {} train {}".format(i,j,gg,hh,ggg))




    #testing

    while True:

        user_input = str(input())
        data_sp = user_input.split()
        final_data = []
        for i in data_sp:
            try:
                final_data.append(np.array(words_y).tolist().index(i.lower()))
            except ValueError:
                final_data.append(399999)
        a, b, c, d, r = sess.run([model, concat_out, logi_ts, prob, pred], feed_dict={input_x: [final_data]})

        print(d, r)










