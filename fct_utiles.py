import os
import zipfile
import requests
from PIL import Image
import numpy as np
import sys
import csv
import shutil


# Définition des variables
DATAS_LOCAL_PATH = './DATAS/'
RAW_LOCAL_PATH = DATAS_LOCAL_PATH + 'RAW/'
ZIP_LOCAL_PATH = RAW_LOCAL_PATH + 'cifar-100.zip'
CURATED_LOCAL_PATH = DATAS_LOCAL_PATH + 'CURATED/'
DATASET_TRAIN_PATH = CURATED_LOCAL_PATH + 'dataset_train.csv'
DATASET_TEST_PATH = CURATED_LOCAL_PATH + 'dataset_test.csv'
MODELS_LOCAL_PATH = './MODELS/'
URL = 'https://stdatalake010.blob.core.windows.net/public/cifar-100.zip'



def check_folder ():
    PATH = [DATAS_LOCAL_PATH, RAW_LOCAL_PATH, CURATED_LOCAL_PATH, MODELS_LOCAL_PATH]
    for p in PATH:
        if not os.path.exists(p):
            os.mkdir(p)


def ensure_data_loaded():
    '''
    Ensure if data are already loaded. Download if missing
    '''
    check_folder()

    if os.path.exists(ZIP_LOCAL_PATH) == False:
        dl_data()
    else :
        print('Datas already downloaded.')

    if os.path.exists(f'{RAW_LOCAL_PATH}cifar-100/') == False:
        extract_data()

    print ('Datas are successfully loaded.\n')


def dl_data ():
    print ('Downloading...')
    with open(ZIP_LOCAL_PATH, "wb") as f:
        r = requests.get(URL)
        f.write(r.content)
    print ('Dataset dowloaded successfully.')


def extract_data():
    print (f'Extracting...')
    with zipfile.ZipFile(ZIP_LOCAL_PATH, 'r') as z:
        z.extractall(f'{RAW_LOCAL_PATH}cifar-100/') 

    print ('Successfull.')


def copy_data(objets):
    itempathtest = []
    itempathtrain = []

    shutil.rmtree(CURATED_LOCAL_PATH + 'test/')
    shutil.rmtree(CURATED_LOCAL_PATH + 'train/')
    
    for objet in objets :
        testpathsource = RAW_LOCAL_PATH + 'cifar-100/test/' + objet + '/'
        trainpathsource = RAW_LOCAL_PATH + 'cifar-100/train/' + objet + '/'
        testpathdest = CURATED_LOCAL_PATH + 'test/' + objet + '/'
        trainpathdest = CURATED_LOCAL_PATH + 'train/' + objet + '/'

        itempathtest.append(testpathdest)
        itempathtrain.append(trainpathdest)

        shutil.copytree(testpathsource, testpathdest)       
        shutil.copytree(trainpathsource, trainpathdest)


    print ('Echantillon copié dans dossier CURATED.')


def png_to_csv (liste, path, step) :

    form='.png'
    fileList = []

    if step == 'train':
        DATASET_PATH = DATASET_TRAIN_PATH
        first_index_label = 22

    else :
        DATASET_PATH = DATASET_TEST_PATH
        first_index_label = 21



    for directory in path :
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                if name.endswith(form):
                    fullName = f'{root}{name}'
                    fileList.append(fullName)
    
    if os.path.exists(DATASET_PATH) :
        os.remove(DATASET_PATH)


    with open(DATASET_PATH, 'a') as f:
        for filename in fileList:

            item = filename[-8:-4]
            label = liste.index(filename[first_index_label:-9])

            img_file = Image.open(filename)

            value = np.asarray(img_file.getdata(),dtype=np.int).reshape((img_file.size[1],img_file.size[0], 3))
            value = np.insert(value, 0, label)
            value = value.flatten()

            with open(DATASET_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(value)
    
    print ('All png files convert to a csv file.')