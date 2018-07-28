# coding: utf-8
#preprocessing third step
#removing punctuations , creating vocabulary and encoding the data

import pickle as pk
import random
from tqdm import tqdm
import sys
import sys
import unicodedata
import nltk


with open('data_all_zipped2.pkl','rb') as f:
    file_read=pk.load(f)



def counting_labels(file_read):
    """
       counting # of yes samples and # of no samples
    :param
       file_read:

    :return:
       total yes labels
       total no labels
       total
    """
    print("counting_labels")
    yes_labels = []
    no_labels = []
    for i in tqdm(file_read):
        if i[0] == 'Y':
            yes_labels.append(i)
        elif i[0] == 'N':
            no_labels.append(i)

    return "total yes labels: {} total no labels: {} total {}".format(len(yes_labels), len(no_labels),
                                                                      len(yes_labels) + len(no_labels))
    # print(counting_labels(file_read))

print(counting_labels(file_read))


def remove_pun(text):
    """
      remove punctuations
    :param
       text:

    :return:
       text without any punctuation
    """
    punctuation = dict.fromkeys([i for i in range(sys.maxunicode)
                                 if unicodedata.category(chr(i)).startswith('P')])

    # removing punctuation
    return [i.lower() for i in nltk.word_tokenize(text.translate(punctuation))]


def vocab_():
    """
    creating vocabulary for model

    :return:
       vocab len
       cleaned data len
    """
    cleaned_data = []
    vocab = set()
    labels = {'Y': 1, 'N': 0}
    for i in tqdm(file_read):
        cleaned_data.append((remove_pun(i[1]), labels[i[0]]))
        vocab.update(remove_pun(i[1]))
    data_vocab = {j: i for i, j in enumerate(list(vocab))}
    with open('dict_vocabulary.pkl', 'wb') as f:
        pk.dump(data_vocab, f)

    with open('cleaned_data_list.pkl', 'wb') as f:
        pk.dump(cleaned_data, f)

    return {'vocab_len': len(vocab), 'cleaned_data': len(cleaned_data)}


# sometimes I believe compiler ignores all my comments :/
#  ヘ( ^o^)ノ＼(^_^ ) high five if you got that

print(vocab_())


with open('cleaned_data_list.pkl', 'rb') as f:
    clean_data = pk.load(f)

with open('dict_vocabulary.pkl', 'rb') as f:
    vocab_dict = pk.load(f)

def encoding_data():
    """
    encode the data using vocabulary
    :return:
      encoded data

    """
    encoded_data = []
    for i in tqdm(clean_data):
        encoded_data.append(([vocab_dict[k] for k in i[0]], i[1]))

    no_dub = set(map(tuple, [i[0] + [i[1]] for i in tqdm(encoded_data)]))

    with open('encoded_data_list.pkl', 'wb') as f:
        pk.dump(no_dub, f)

    

    return len(encoded_data)

print(encoding_data())




