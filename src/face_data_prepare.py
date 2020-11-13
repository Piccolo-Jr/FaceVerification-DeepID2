import os
import shutil
import cv2
import numpy as np
import random
import pickle


def vec_face_dataset(train_path,file_path):

    train_data, train_label, valid_data, valid_label = [], [], [], []
    count = 0
    for folder in os.listdir(train_path):
        print(count)
        img_list = os.listdir(os.path.join(train_path,folder,'face'))
        for img_name in img_list[:-2]:
            img = cv2.imread(os.path.join(train_path,folder,'face',img_name))
            train_data.append(np.asarray(cv2.resize(img,(47, 55)),dtype='float32'))
            train_label.append(count)
        for img_name in img_list[-2:]:
            img = cv2.imread(os.path.join(train_path,folder,'face',img_name))
            valid_data.append(np.asarray(cv2.resize(img,(47, 55)),dtype='float32'))
            valid_label.append(count)
        count += 1
    train_data = np.asarray(train_data, dtype='float32')
    train_label = np.asarray(train_label, dtype='int32')
    valid_data = np.asarray(valid_data,dtype='float32')
    valid_label = np.asarray(valid_label,dtype='int32')
    print(train_data.shape,train_label.shape,valid_data.shape,valid_label.shape)

    train_face_X_pkl = pickle.dumps(train_data,pickle.HIGHEST_PROTOCOL)
    train_face_Y_pkl = pickle.dumps(train_label,pickle.HIGHEST_PROTOCOL)
    valid_face_X_pkl = pickle.dumps(valid_data,pickle.HIGHEST_PROTOCOL)
    valid_face_Y_pkl = pickle.dumps(valid_label,pickle.HIGHEST_PROTOCOL)
    del train_data, train_label, valid_data, valid_label

    max_bytes = 2**31 - 1
    with open(os.path.join(file_path,'train_face_X.pkl'),'wb') as file:
        for idx in range(0, len(train_face_X_pkl), max_bytes):
            file.write(train_face_X_pkl[idx:idx + max_bytes])
    with open(os.path.join(file_path,'train_face_Y.pkl'),'wb') as file:
        for idx in range(0, len(train_face_Y_pkl), max_bytes):
            file.write(train_face_Y_pkl[idx:idx + max_bytes])
    with open(os.path.join(file_path,'valid_face_X.pkl'),'wb') as file:
        for idx in range(0, len(valid_face_X_pkl), max_bytes):
            file.write(valid_face_X_pkl[idx:idx + max_bytes])
    with open(os.path.join(file_path,'valid_face_Y.pkl'),'wb') as file:
        for idx in range(0, len(valid_face_Y_pkl), max_bytes):
            file.write(valid_face_Y_pkl[idx:idx + max_bytes])


if __name__ == '__main__':

    vec_face_dataset('../data/CelebFaces+A','../file/CelebFaces+')