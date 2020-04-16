import random
import numpy as np
import math
import pickle
import time
import datetime
import pandas as pd
import os
from Network import Network
from Dataset import Dataset
from Config import *
import Util
from sklearn.model_selection import train_test_split


# get the training data for a specific fold
def getTrainingData(allFolds,fold):
    data_df = pd.DataFrame.from_dict({"SentenceId":[]})
    for i, thisFold in enumerate(allFolds):
        if (i != fold):
            data_df=data_df.append(allFolds[i]);

    # pre-defined set of columns
    classes=["negative","positive"]

    return  data_df,classes


# get full training data
def getFullTrainingData(allFolds):
    data_df = pd.DataFrame.from_dict({"SentenceId":[]})
    for i, thisFold in enumerate(allFolds):
        data_df=data_df.append(allFolds[i]);

    # pre-defined set of columns
    classes=["negative","positive"]

    return data_df,classes


def runClassificationExperimentKPartition(foldSize, start, maxK, step):

    print('run experiments for '+Config.NETWORK)

    allFoldK = pickle.load(open('allFold_2019.pickle', "rb"))
    allTestData = pickle.load(open('allTestData_2019.pickle', "rb"))
    allSentences = pickle.load(open('sentences_2019.pickle', "rb"))

    sentences_Secion = pickle.load(open('sentences_Secion_Label_2019.pickle', "rb"))
    sentences_Secion.columns=['id', 'section']


    for k in range(start,maxK+1,step):
        for f in range(foldSize):
            print("K = ", k, " F = ", f);
            if os.path.isfile("results_" + str(k) + "_" + str(f)+ "_TEST"+str(Config.TEST) + ".pickle"):
                print('found ', "results_" + str(k) + "_" + str(f)+ "_TEST"+str(Config.TEST) + ".pickle")
            else:
                [training_df,classes]  = getTrainingData(allFoldK[k],f)

                test_df = allTestData[f]
                sentences= allSentences[allSentences['id'].isin(training_df['SentenceId'])]
                training_df=training_df.rename(columns={'SentenceId':'id'})
                test_df = test_df.rename(columns={'SentenceId': 'id'})
                training_df=pd.merge(training_df, sentences, on='id')
                training_df = pd.merge(training_df, sentences_Secion, on='id')
                test_df = pd.merge(test_df, sentences_Secion, on='id')
                section_classes=None
                if (Config.USE_SECTION):
                    section_classes = np.unique(training_df['section'].tolist()+test_df['section'].tolist())

                training_df, validation_df = train_test_split(training_df , test_size=Config.TRAIN_TEST_RATIO, stratify=training_df[classes].values,
                                                                                     random_state=42)

                dataset=Dataset(training_df,validation_df,test_df,classes,section_classes,k, preTrain=True)

                model = Network()
                if (os.path.isfile("model_" + str(k) + "_" + str(f) + ".h5")):

                    model.loadModel("model_" + str(k) + "_" + str(f) + ".h5")
                else:
                    model.build(dataset)
                    model.run(dataset)
                    model.saveModel("model_" + str(k) + "_" + str(f) + ".h5")

                ids, preds=model.predict(dataset)
                results = []
                for i, pred in enumerate(preds):
                    results.append(['HAN', k, f, dataset.X_test[i][0],
                                    pred,
                                    dataset.get_Y_test()[i],
                                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')])

                pickle.dump(results, open("results_" + str(k) + "_" + str(f) + "_TEST"+str(Config.TEST)+ ".pickle", "wb"))
                weights = model.getActivationWeights(dataset)
                pickle.dump(weights,
                            open("weights_" + str(k) + "_" + str(f) + "_TEST" + str(Config.TEST) + ".pickle", "wb"))


def runITHSExperiment(k):

    print('run experiments for ' + Config.NETWORK)

    allFoldK = pickle.load(open('allFold_2019.pickle', "rb"))
    allTestData = Util.readData( 'sentences_all_2019.pickle' )
    allTestData = allTestData.iloc[0:4000000, :]
    allSentences = pickle.load(open('sentences_2019.pickle', "rb"))

    [training_df, classes] = getFullTrainingData(allFoldK[k])
    test_df = allTestData
    sentences = allSentences[allSentences['id'].isin(training_df['SentenceId'])]
    training_df = training_df.rename(columns={'SentenceId': 'id'})
    training_df = pd.merge(training_df, sentences, on='id')

    training_df, validation_df = train_test_split(training_df, test_size=Config.TRAIN_TEST_RATIO,
                                                  stratify=training_df[classes].values,
                                                  random_state=42)

    dataset = Dataset(training_df, validation_df, test_df, classes, None, k,preTrain=False)

    model = Network()
    if (os.path.isfile("model_iths.h5")):
        model.loadModel("model_iths.h5")

    else:
        model.build(dataset)
        model.run(dataset)
        model.saveModel("model_iths.h5")

    ids, preds = model.predict(dataset)
    weights = model.getActivationWeights(dataset)
    pickle.dump((ids ,preds, weights), open('preds_all.pickle', "wb"))




def runITHS():

    Config.NETWORK = 'HAN'
    Config.USE_SECTION = False
    Config.SHUFFLE = False
    Config.USE_METAMAP = False
    Config.TEST = 1
    Config.METAMAP_GRU_UNITS = 100
    Config.CLASS_WEIGHT_NEG = None
    Config.CLASS_WEIGHT_POS = 0.15
    Config.W2V_ITERATION = 15
    Config.W2V_HS = 0
    Config.W2V_WINDOW = 5
    Config.W2V_NEGATIVE = 5
    Config.GRU_UNITS_WORD = 100
    Config.GRU_UNITS_SENT = 300
    Config.GRU_UNITS_DOC = 300
    Config.DROP_OUT = 0.4
    Config.MAX_SENTS = 4
    Config.VALIDATION_SPLIT = 0.0
    Config.TRAIN_TEST_RATIO = 0.2
    Config.PATIENCE = 10
    runITHSExperiment(32)

if __name__ == '__main__':

    runITHS()
