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
DATASET_PATH = CURATED_LOCAL_PATH + 'dataset.csv'
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
    for objet in objets :
        testpathsource = RAW_LOCAL_PATH + 'cifar-100/test/' + objet
        trainpathsource = RAW_LOCAL_PATH + 'cifar-100/train/' + objet
        testpathdest = CURATED_LOCAL_PATH + 'test/' + objet
        trainpathdest = CURATED_LOCAL_PATH + 'train/' + objet

        shutil.copytree(testpathsource, testpathdest)
        shutil.copytree(trainpathsource, trainpathdest)


    print ('Files successfully copied.')


def png_to_csv (car_path, number) :

    fullRawDir = []

    for cp in car_path :
        frd = f'{RAW_LOCAL_PATH}{cp}'
        fullRawDir.append(frd)

    form='.png'
    fileList = []

    for directory in fullRawDir :
        for root, dirs, files in os.walk(directory, topdown=False):
            n = 0
            for name in files:
                if n == number :
                    break
                else :
                    if name.endswith(form):
                        fullName = f'{root}{name}'
                        fileList.append(fullName)
                        n += 1
    
    if os.path.exists(DATASET_PATH) :
        os.remove(DATASET_PATH)

    with open(DATASET_PATH, 'a') as f:
        for filename in fileList:

            lettre = filename[29]
            label = alphabet.index(lettre)

            img_file = Image.open(filename)

            value = np.asarray(img_file.getdata(),dtype=np.int).reshape((img_file.size[1],img_file.size[0]))
            value = np.insert(value, 0, label)
            value = value.flatten()

            with open(DATASET_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(value)
    
    print ('All png files convert to a csv file.')