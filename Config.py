# Set the dimensions of the input and the embedding.
# The input has to be a 3-dimensional tensor of fixed size (sample_size x n_sentences x n_words).
#
# MAX_SENT_LEN : the number of words in each sentence.
#
# MAX_SENTS : the number of sentences in each document.
#
# MAX_NB_WORDS : the size of the word encoding
#
# EMBEDDING_DIM : the dimensionality of the word embedding

import tensorflow as tf
import os
from tensorflow.python.client import device_lib

def get_number_of_gpus():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    local_device_protos = device_lib.list_local_devices()
    gpus =[x.name for x in local_device_protos if x.device_type == 'GPU']
    sess.close()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    return gpus.__len__()


class Config():
    NETWORK='HAN'
    GPU=get_number_of_gpus()

    MAX_SENT_LENGTH = 40
    MAX_SENTS = 3
    MAX_WORD_LENGTH = 10
    MAX_NB_WORDS = 40000
    EMBEDDING_DIM = 300
    EMBEDDING_DIM_WORD = 82
    DROP_OUT=0.5
    RECURRENT_DROPOUT=0.

    GRU_UNITS_WORD=100
    GRU_UNITS_SENT=100
    GRU_UNITS_DOC=100
    SHUFFLE=False
    # parameters to train the network
    # BATCH_SIZE = 256
    # NUM_EPOCHS = 15
    CLASS_WEIGHT_NEG = 0.5
    CLASS_WEIGHT_POS = 0.5

    W2V_ITERATION=15
    W2V_HS=0
    W2V_WINDOW=5
    W2V_NEGATIVE=5

    TRAIN_TEST_RATIO=0.33
    PATIENCE=15
    PRE_TRAIN=True

    REG_PARAM = 1e-7
    VALIDATION_SPLIT=0.1

    USE_METAMAP = True
    MAX_METAMAP = 5000
    MAX_METAMAP_LENGTH=15
    METAMAP_GRU_UNITS = 100

    CONNECT_DB = False
    DB_HOST = 'localhost'
    DB_USER ='rad'

    PLOT=False
    NN_PARAMS={'epochs': [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 1, 1], 'batch_size': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 256, 256, 256, 256, 256, 256, 256, 256,256]}


    @staticmethod
    def init():

        Config.NETWORK = 'HAN'
        Config.GPU = get_number_of_gpus()

        Config.MAX_SENT_LENGTH = 40
        Config.MAX_SENTS = 3
        Config.MAX_WORD_LENGTH = 30
        Config.MAX_NB_WORDS = 40000
        Config.EMBEDDING_DIM = 300
        Config.DROP_OUT = 0.5
        Config.RECURRENT_DROPOUT =None

        Config.GRU_UNITS_WORD = 100
        Config.GRU_UNITS_SENT = 100
        Config.GRU_UNITS_DOC=100
        Config.SHUFFLE = False

        Config.W2V_ITERATION = 15
        Config.W2V_HS = 0
        Config.W2V_WINDOW = 5
        Config.W2V_NEGATIVE=0

        Config.TRAIN_TEST_RATIO = 0.33
        Config.PATIENCE=15
        Config.PRE_TRAIN = True

        Config.REG_PARAM = 1e-7
        Config.VALIDATION_SPLIT = 0.1


        Config.PLOT = False
        Config.NN_PARAMS = {
            'epochs': [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
            'batch_size': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 256, 256, 256, 256, 256, 256, 256, 256]}




