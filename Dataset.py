from Config import Config
import Util
import pandas as pd
import numpy as np
import nltk
import pickle

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import   text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Dataset():

    def __init__(self, training_df,validation_df,test_df,classes, section_classes, k, preTrain=Config.PRE_TRAIN):

        self.k=k
        self.training_df = training_df
        self.X_train = training_df[['id', 'text']].values
        self.Y_train = training_df[classes].values
        train_data = pd.DataFrame(data={'text': self.X_train[:, 1].tolist(), 'label': self.Y_train.tolist()})
        train_docs = []
        train_labels = []
        train_texts = []
        self.__preprocess(train_data, train_docs, train_labels, train_texts)

        corpusTexts = []
        corpusDocs = []
        corpusTexts.extend(train_texts)
        corpusDocs.extend(train_docs)

        if (preTrain == True):
            allTexts_ITHS = Util.readData("pretrain_iths_texts.pickle")
            allDocs_ITHS = Util.readData("pretrain_iths_docs.pickle")
            allTexts, allDocs = pickle.load(open('pretrain.pickle', "rb"))
            texts = allTexts + allTexts_ITHS
            docs = allDocs + allDocs_ITHS

            corpusTexts = texts
            corpusDocs = docs
            print('PRE TRAIN texts')

        print('corpus size: ', len(corpusDocs))

        self.n_classes = len(classes)

        # HAN
        if(Config.NETWORK == 'HAN'):
            from keras.preprocessing.text import Tokenizer
            self.__tokenizer = Tokenizer(num_words=Config.MAX_NB_WORDS, lower=False)
            self.__tokenizer.fit_on_texts(corpusTexts)

            [_, _, self.doc_lst] = self.__tokenize(corpusTexts, corpusDocs)
            [self.han_x_train,_, _] = self.__tokenize(train_texts, train_docs)

            self.word_index = self.__tokenizer.word_index
            print('Total %s unique tokens.' % len(self.word_index))

            self.han_y_train = np.asarray(train_labels)

            print('Shape of data tensor:', self.han_x_train.shape)
            print('Shape of label tensor:', self.han_y_train.shape)


        # section
        if(Config.USE_SECTION):
            encoder = LabelBinarizer()
            encoder.fit(section_classes)
            self.section_x_train = encoder.transform(training_df['section'])
            self.section_classes = section_classes

        self.test_df = test_df

        if validation_df is not None:
            self.X_validation = validation_df[['id', 'text']].values
            if all(c in validation_df for c in classes):
                self.Y_validation = validation_df[classes].values
                validation_data = pd.DataFrame(data={'text': self.X_validation[:, 1].tolist(), 'label': self.Y_validation.tolist()})
            else:
                validation_data = pd.DataFrame(data={'text': self.X_validation[:, 1].tolist(), 'label': None})

            validation_docs = []
            validation_labels = []
            validation_texts = []
            print('preprocessing validation data....')
            self.__preprocess(validation_data, validation_docs, validation_labels, validation_texts)

            print('tokenizing validation data....')
            if (Config.NETWORK == 'HAN'):
                [self.han_x_validation, self.han_x_validation_doc, _] = self.__tokenize(validation_texts, validation_docs)
                self.han_y_validation = np.asarray(validation_labels)

            if (Config.USE_SECTION):
                self.section_x_validation = encoder.transform(validation_df['section'])

        if test_df is not None:
            self.X_test = test_df[['id', 'text']].values
            if all(c in test_df for c in classes):
                self.Y_test = test_df[classes].values
                test_data = pd.DataFrame(data={'text': self.X_test[:, 1].tolist(), 'label': self.Y_test.tolist()})
            else:
                test_data = pd.DataFrame(data={'text': self.X_test[:, 1].tolist(), 'label': None})

            test_docs = []
            test_labels = []
            test_texts = []
            print('preprocessing test data....')
            self.__preprocess(test_data, test_docs, test_labels, test_texts)

            print('tokenizing test data....')
            if (Config.NETWORK == 'HAN'):
                [self.han_x_test, self.han_x_test_doc, _] = self.__tokenize(test_texts, test_docs)
                self.han_y_test = np.asarray(test_labels)

            if (Config.USE_SECTION):
                self.section_x_test = encoder.transform(test_df['section'])


    def get_X_train(self):
        if(Config.NETWORK == 'HAN'):
            return self.han_x_train

    def get_Y_train(self):
        if(Config.NETWORK == 'HAN'):
            return self.han_y_train

    def get_X_validation(self):
        if(Config.NETWORK == 'HAN'):
            return self.han_x_validation

    def get_Y_validation(self):
        if(Config.NETWORK == 'HAN'):
            return self.han_y_validation

    def get_X_test(self):
        if(Config.NETWORK == 'HAN'):
            return self.han_x_test

    def get_Y_test(self):
        if(Config.NETWORK == 'HAN'):
            return self.han_y_test

    # preprocess text to prepare for training
    def __tokenize(self, texts, docs):
        data = np.zeros((len(texts), Config.MAX_SENTS, Config.MAX_SENT_LENGTH), dtype='int32')
        doc = []
        doc_lst = []

        for i, sentences in enumerate(docs):
            sents_in_doc=[]
            for j, sent in enumerate(sentences):
                if j < Config.MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent, lower=False,filters='!"#$%&()*+,:;<=>?@[\\]^_`{|}~\t\n')
                    k = 0
                    words_in_sent = []
                    originals_words_in_sent=[]
                    for _, word in enumerate(wordTokens):
                        originals_words_in_sent.append(word)
                        if k < Config.MAX_SENT_LENGTH:
                            if (word in self.__tokenizer.word_index) and (self.__tokenizer.word_index[word] < Config.MAX_NB_WORDS):
                                data[i, j, k] = self.__tokenizer.word_index[word]
                                words_in_sent.append(word)
                            else:
                                data[i, j, k] = Config.MAX_NB_WORDS
                                words_in_sent.append('UNK')
                            k = k + 1

                    doc_lst.append(words_in_sent)
                    if (originals_words_in_sent.__len__()>0):
                        sents_in_doc.append(originals_words_in_sent)
                    else:
                        sents_in_doc.append(sent)

            doc.append(sents_in_doc)

        return [data,doc, doc_lst]


    def __preprocess(self, data, docs, labels, texts ):

        for idx in range(data.shape[0]):
            text = data['text'].iloc[idx] #.replace('\n',' ')
            texts.append(text)
            sentences = nltk.tokenize.sent_tokenize(text)
            docs.append(sentences)
            if(labels is not None):
                labels.append(data['label'].iloc[idx])

    def getWordFromIndex(self,index):
        return self.__tokenizer.index_word[index]
