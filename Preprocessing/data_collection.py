# coding: utf-8

#preprocessing first step
#collecting the data from sources


import os
import time




#download the data files from 1998 to 2018 
text='https://www.accessdata.fda.gov/MAUDE/ftparea/foitext2001.zip'
text_mdr='https://www.accessdata.fda.gov/MAUDE/ftparea/mdrfoithru2017.zip'

def download_data_files():

    """
    :return:
       downloaded files names

    """
    data=text.split('/')

    #download all zip files
    for i in range(1998,2005):
        data[5]='foitext' + str(i) + '.zip'
        os.system("wget " + "/".join(data))

    os.system("wget " + text_mdr)
    os.system('unzip mdrfoithru2017.zip')

    #unzip all zip files
    for i in range(1998, 2005):
        os.system('unzip foitext' + str(i) + '.zip')




    #delete all .zip files
    for i in os.listdir('./'):
        if '.zip' in i:
            os.system('rm -rf ' + str(i))






    return os.listdir('./')

#unzip the files

# download_data_files()






