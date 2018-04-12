import re
import sys
import os
import numpy as np
import unicodedata

dir_a='/Users/exepaul/Desktop/train/'


final_ = []

data_x=np.load('word_embedding_lstm.npy')
words_y=np.load('words_list_lstm.npy')


def get_final(data_line):
    sub_final = []
    fg = " ".join(re.split(r'\/|-', data_line))
    data_line=fg.replace('<br  >', ' ')


    d = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    new = data_line.translate(d).split()
    print(data_line)
    print(new)
    for i in new:
        try:
            sub_final.append(np.array(words_y).tolist().index(i.lower()))
        except ValueError:
            sub_final.append(399999)
    final_.append(sub_final)



all_pos_files=os.listdir(dir_a+'pos')

all_neg_files=os.listdir(dir_a+'neg')




positive_rev=[]
negative_rev=[]

for i in all_pos_files:
    with open(dir_a+'pos/'+i) as f:
        data=f.readlines()
        positive_rev.append(data[0])

for i in all_neg_files:
    with open(dir_a+'neg/'+i) as f:
        data=f.readlines()
        negative_rev.append(data[0])


print(len(positive_rev))
