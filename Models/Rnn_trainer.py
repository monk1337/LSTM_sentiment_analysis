# coding: utf-8

#training file for LSTM rnn
import random
import time
import os 
import numpy as np
import pickle as pk
import tensorflow as tf
from code_network import LSTM_Network


epoch = 10





with open('../Preprocessing/encoded_data_list.pkl', 'rb') as f:
    file_data = pk.load(f)
#Shuffled the data
file_data = list(file_data)
random.shuffle(file_data)
print(len(file_data))

#split the data into train and test dataset
train_data = int(len(file_data) * 0.85)
train_data_in = file_data[:train_data]  # split train_data
test_data_in = file_data[train_data:]  # split test data

#sorted for minimal padding
sorted_train_data = sorted(train_data_in, key=lambda x: len(x))[2:]

sorted_test_data = sorted(test_data_in, key=lambda x: len(x))
#saving the test for validation
np.save('sorted_test_data', sorted_test_data)

print("test_data", len(sorted_train_data))
print("train_data", len(sorted_test_data))






def sequence_padding(seqs_):

    """
    Padding the seq with same length for RNN

    :param seqs_:
           taking batch of without padded seqs as argument  ex :  [[278698,3442], [194661] , [1098,2341,77]]

    :return:
           padded  with zeros  with max len of seq in batch   ex : [ [278698,3442,0] , [194661,0,0] ,[1098,2341,77]]

    :raise:
          if seq is not 2 dimension:
            TypeError: object of type 'int' has no len()

    """
    max_sequence = max(list(map(len, seqs_)))

    # getting Max length of sequence
    padded_sequence = [i + [0] * (max_sequence - len(i)) if len(i) < max_sequence else i for i in seqs_]

    # padded_sequence = list(map(lambda x:x + [0]*(max_sequence-len(x)) if len(x)<max_sequence else x,seqs))
    # padded sequence with max length of sequence
    return padded_sequence
    # return padded sequence


#
#
def reshape_data(batch_seq_data):

    """
    Reshaping the sequences with np array format

    :param batch_seq_data:
           taking batch of seq as argument

    :return:
           batch of np array values
    """


    # inputs = padding([list(i[:-1]) for i in batch_seq_data], hj=j)
    sentence_data = sequence_padding([list(i[:-1]) for i in batch_seq_data])
    # print([list(i[:-1]) for i in batch_seq_data])

    labels = [int(i[-1]) for i in batch_seq_data]

    return {

        'input': np.array(sentence_data),
        'labels': np.array(labels)
    }


def evaluate_(model, batch_size=50):

    """
    Checking  the test accuracy on testing data set

    :param model:
       current lstm model

    :param batch_size:
       batch size for test data set

    :return:
       mean accuracy of test data set

    :raise:
       if input shape is different from placeholder shape:
          ValueError: Cannot feed value of shape

    """
    sess = tf.get_default_session()

    # batch_data = test_data_in
    # batch_labesl = train_labels

    iteration = len(sorted_test_data) // batch_size

    accuracy = []

    for i in range(iteration):
        batch_data_i = sorted_test_data[i * batch_size:(i + 1) * batch_size]

        input_test = reshape_data(batch_data_i)['input']
        labels = reshape_data(batch_data_i)['labels']

        network_out = sess.run(model.output, feed_dict={model.placeholder['input']: input_test,
                                                        model.placeholder['label_s']: labels,
                                                        model.placeholder['dropout_mode']: 1})

        accuracy.append(network_out['accuracy'])
    return np.mean(np.array(accuracy))


def train_model(model, batch_size=350):
    """

    :param model:
          current lstm model

    :param batch_size:
          batch size for training

    :print:
          epoch
          iteration
          training_loss
          accuracy

    """
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        iteration = len(sorted_train_data) // batch_size
        print("iteration", iteration)
        time.sleep(5)

        for i in range(epoch):

            for j in range(iteration):
                batch_data_j = sorted_train_data[j * batch_size:(j + 1) * batch_size]

                input_train = reshape_data(batch_data_j)['input']

                labels = reshape_data(batch_data_j)['labels']

                network_out, _ = sess.run([model.output, model.train],
                                          feed_dict={model.placeholder['input']: input_train,
                                                     model.placeholder['label_s']: labels,
                                                     model.placeholder['dropout_mode']: 0})

                print({'epoch': i,
                       'iteration': j,
                       'training_loss': network_out['loss'],
                       'training_accuracy': network_out['accuracy']
                       })

                if j % 500 == 0:
                    with open('iterres.txt', 'a') as f:
                        f.write(
                            str({'epoch': i, 'test_accuracy': evaluate_(model, batch_size=100), 'iteration': j}) + '\n')
                        os.system('mkdir ' + str(i) + 'epoch' + str(j))
                        saver.save(sess, '/home/ANANT/apal/' + str(i) + 'epoch' + str(j) + '/' + str(i))

            print({'epoch': i, 'test_accuracy': evaluate_(model)})
            with open('epochandresult', 'a') as f:
                f.write(str({'epoch': i, 'test_accuracy': evaluate_(model)}) + '\n')
            os.system('mkdir ' + str(i) + 'epoch')
            saver.save(sess, '/home/ANANT/apal/' + str(i) + 'epoch' + '/' + str(i))


if __name__ == "__main__":
    model = LSTM_Network(351360, 0.5, 300, 1.0, 200)

    train_model(model)
