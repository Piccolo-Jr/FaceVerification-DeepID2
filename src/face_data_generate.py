import os
import pickle
import numpy as np
import tensorflow as tf


def load_train_data():

    max_bytes = 2**31 - 1
    input_size = os.path.getsize('../file/CelebFaces+/valid_face_X.pkl')
    with open('../file/CelebFaces+/valid_face_X.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        validX = pickle.loads(bytes_in)

    input_size = os.path.getsize('../file/CelebFaces+/valid_face_Y.pkl')
    with open('../file/CelebFaces+/valid_face_Y.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        validY = pickle.loads(bytes_in)
    
    input_size = os.path.getsize('../file/CelebFaces+/train_face_X.pkl')
    with open('../file/CelebFaces+/train_face_X.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        trainX = pickle.loads(bytes_in)

    input_size = os.path.getsize('../file/CelebFaces+/train_face_Y.pkl')
    with open('../file/CelebFaces+/train_face_Y.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        trainY = pickle.loads(bytes_in)

    new_trainX = np.asarray([trainX[i:i+2] for i in range(0,len(trainX)-1,2)],dtype='float32')
    new_trainY = np.asarray([trainY[i:i+2] for i in range(0,len(trainY)-1,2)],dtype='int32')

    new_validX = np.asarray([validX[i:i+2] for i in range(0,len(validX)-1,2)],dtype='float32')
    new_validY = np.asarray([validY[i:i+2] for i in range(0,len(validY)-1,2)],dtype='int32')

    
    return (new_trainX, new_trainY), (new_validX, new_validY)


def load_test_data():

    max_bytes = 2**31 - 1
    input_size = os.path.getsize('../file/LFW/testX1.pkl')
    with open('../file/LFW/testX1.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        testX1 = pickle.loads(bytes_in)

    input_size = os.path.getsize('../file/LFW/testX2.pkl')
    with open('../file/LFW/testX2.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        testX2 = pickle.loads(bytes_in)

    input_size = os.path.getsize('../file/LFW/testY.pkl')
    with open('../file/LFW/testY.pkl', 'rb') as f:
        bytes_in = bytearray(0)
        for idx in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
        testY = pickle.loads(bytes_in)

    return testX1, testX2, testY


if __name__ == '__main__':
    t1 = [1,2,3,4,5,6]
    t2 = [11,22,33,44,55,66]

    print(tf.transpose(tf.concat([[t1],[t2]], 0)))


