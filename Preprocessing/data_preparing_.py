# coding: utf-8
#second step in prepossessing
#preparing the cleaned data from raw_data and combining all files into one final file

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pk
import os
print('libraries loaded')


def load_labels():
    """
    Load the mdrfoiThru2017.txt from url , file contains Mdr_report_key along with adverse event label
    :return:
        mdrfoiThru2017 pandas dataframe

    :raises:
        FileNotFoundError: if file is not in current directory or file name is wrong
    """

    return pd.read_csv("mdrfoiThru2017.txt", sep='|', encoding='latin1', error_bad_lines=False)




def merged_and_combine(raw_file_name, data_labels):
    """
        load the text file and label file , combine both file basis on mrd_report_key
        drop the NaN or empty labels / text from file
        taking 60000 samples from each file
    :param
        raw_file_name

    :param
         data_labels

    :return:
        stored file name
        slide file name

    :raises:
       if wrong column name then:
          "[] not in index"
    """
    print(raw_file_name)
    slide_name = str('final') + raw_file_name  # delete
    df_text = pd.read_csv(raw_file_name, sep='|', encoding='latin1', error_bad_lines=False)  # change

    df_mdrfoi = data_labels[['ADVERSE_EVENT_FLAG', 'MDR_REPORT_KEY']]
    df_mdrfoi['ADVERSE_EVENT_FLAG'].replace('', np.nan, inplace=True)

    df_text = df_text[['MDR_REPORT_KEY', 'FOI_TEXT']]
    df_text['FOI_TEXT'].replace('', np.nan, inplace=True)

    df_med_ae_data = pd.merge(df_mdrfoi, df_text, on=['MDR_REPORT_KEY'])
    df_med_ae_data.dropna(subset=['ADVERSE_EVENT_FLAG', 'FOI_TEXT'], inplace=True)

    data = df_med_ae_data[:60000]
    np.save(slide_name, data)  # change


    return {
               'file_name': str(slide_name) + '.npy',
               'slide_name': str('final') + raw_file_name
          }


def slice_and_clean(file_data):
    """
    :param
        file_data:

    :return:
        file_ name
        yes labels
        no labels
        sliced samples
        remain samples ( is it less than 60000 ? )
    """
    data = np.load(file_data['file_name'])

    yes_columns = []
    no_columns = []
    print('slicing and cleaning..',file_data['file_name'])
    for i in tqdm(data):
        if i[0] == 'Y':
            yes_columns.append(i)
        elif i[0] == 'N':
            no_columns.append(i)

    slice_op = yes_columns[:40000] + no_columns[:40000]

    np.save(str(file_data['file_name'][:17] + str('m')), slice_op)  # change
    os.system(str('rm ') + str(file_data['slide_name']) + '.npy')
    return "file_ {} yes {}  no {} sliced {} remain {}".format(str(file_data['file_name'][:17] + str('2m')) + '.npy',
                                                               len(yes_columns), len(no_columns),
                                                               len(slice_op),
                                                               60000 - len(slice_op))
print('loading_labels..')
data_labels = load_labels()
print('labels_loaded')

print('merged and combined process starting')
for i in tqdm(range(1998,2005)):
    print(slice_and_clean(merged_and_combine('foitext' + str(i) + '.txt',data_labels)))


def combined_():
    """
      concat all the sliced_cleaned files into one file
    :return:
       final combined file name

    """
    data_all_zipped = []
    all_len = os.listdir('./')
    size = 0
    print('concatenating all files')
    for i in tqdm(all_len):
        if '.npy' in i:
            data_open = np.load(i)
            for i in tqdm(data_open):
                data_all_zipped.append([i[0], i[2]])

    with open('data_all_zipped2.pkl', 'wb') as f:
        pk.dump(data_all_zipped, f)

    return "all_file_combined{} ".format(len(data_all_zipped))



print(combined_())