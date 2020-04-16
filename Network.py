from Config import Config
import numpy as np
import os 
os.environ['KERAS_BACKEND'] = 'tensorflow'
import time
import gensim, logging, pickle
from keras.layers import Dense, Input, Flatten, Concatenate, Reshape,concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding,   Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, optimizers, layers
from keras.callbacks import History, CSVLogger, EarlyStopping, Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Network():
    def __init__(self):
        self.__embedding_layer = None
        self.__model= None
        self.__parallelmodel= None

        K.clear_session()

    def clear(self):
        K.clear_session()


    def __trainEmbedding(self, dataset):
        print('Training Embedding...' )
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
        if(not os.path.isfile("word2vec.model")):
            # use skip-gram
            word2vec_model = gensim.models.Word2Vec(dataset.doc_lst, min_count=3, window=Config.W2V_WINDOW, iter=Config.W2V_ITERATION, size=Config.EMBEDDING_DIM, sg=1, hs=Config.W2V_HS, negative=Config.W2V_NEGATIVE, workers=os.cpu_count(), seed=43)
            word2vec_model.save("word2vec.model")
        else:
            word2vec_model = gensim.models.Word2Vec.load("word2vec.model")

        embeddings_index = {}

        for word in word2vec_model.wv.vocab:
            coefs = np.asarray(word2vec_model.wv[word], dtype='float32')
            embeddings_index[word] = coefs

        print('Total %s word vectors.' % len(embeddings_index))

        # Initial embedding
        embedding_matrix = np.zeros((Config.MAX_NB_WORDS + 1, Config.EMBEDDING_DIM))

        for word, i in dataset.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None and i < Config.MAX_NB_WORDS:
                embedding_matrix[i] = embedding_vector
            elif i == Config.MAX_NB_WORDS:
                embedding_matrix[i] = embeddings_index['UNK']

        l2_reg = regularizers.l2(Config.REG_PARAM)

        embedding_layer = Embedding(Config.MAX_NB_WORDS + 1,
                                    Config.EMBEDDING_DIM,
                                        input_length=Config.MAX_SENT_LENGTH,
                                        trainable=True,
                                        mask_zero= True,
                                        embeddings_regularizer=l2_reg,
                                        weights=[embedding_matrix])

        return embedding_layer


    def build(self,dataset):

        if(Config.NETWORK=='HAN'):
            self.build_HAN(dataset)


    # building HAN
    def build_HAN(self, dataset):

        GRU_IMPL = 1 if Config.GPU==0 else 2

        # Define the network.
        l2_reg = regularizers.l2(Config.REG_PARAM)

        embedding_layer = self.__trainEmbedding(dataset)

        sentence_input = Input(shape=(Config.MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(Config.GRU_UNITS_SENT, return_sequences=True, kernel_regularizer=l2_reg,
                                   implementation=GRU_IMPL))(embedded_sequences)
        l_lstm = Dense( 100, activation='relu',  name='dense_sentence', kernel_regularizer=l2_reg)(l_lstm)
        l_att = AttLayer(name='word_attention_layer', regularizer=l2_reg)(l_lstm)
        self.__sentEncoder = Model(sentence_input, l_att)

        if (Config.USE_SECTION):
            aux_input = Input(shape=(len(dataset.section_classes),), name='aux_input')
            section_preds = Dense(8, activation='relu', kernel_regularizer=l2_reg)(aux_input)
            section_preds = Dense(4, activation='relu', kernel_regularizer=l2_reg)(section_preds)
            section_preds = Dropout(Config.DROP_OUT)(section_preds)

            doc_input = Input(shape=(Config.MAX_SENTS, Config.MAX_SENT_LENGTH), dtype='int32', name='main_input')
            doc_encoder = TimeDistributed(self.__sentEncoder)(doc_input)

        else:
            doc_input = Input(shape=(Config.MAX_SENTS, Config.MAX_SENT_LENGTH), dtype='int32')
            doc_encoder = TimeDistributed(self.__sentEncoder)(doc_input)

        l_lstm_sent = Bidirectional(GRU(Config.GRU_UNITS_DOC, return_sequences=True, kernel_regularizer=l2_reg,
                                        implementation=GRU_IMPL))(doc_encoder)
        l_att_sent = AttLayer(regularizer=l2_reg)(l_lstm_sent)
        l_att_sent = Dropout(Config.DROP_OUT)(l_att_sent)

        if (Config.USE_SECTION):
            preds = Dense(8, activation='relu', kernel_regularizer=l2_reg)(l_att_sent)
            preds = Dense(4, activation='relu', kernel_regularizer=l2_reg)(preds)
            preds = layers.concatenate([preds, section_preds])
            preds = Dense(dataset.n_classes, activation='softmax', kernel_regularizer=l2_reg)(preds)
        else:
            preds = Dense(dataset.n_classes, activation='softmax', kernel_regularizer=l2_reg)(l_att_sent)

        if (Config.GPU>0):
            with K.tf.device('/cpu:0'):
                if (Config.USE_SECTION):
                    self.__model = Model(inputs=[doc_input, aux_input], outputs=preds)
                else:
                    self.__model = Model(doc_input, preds)


            self.__model = multi_gpu_model(self.__model, gpus=Config.GPU)

        else:
            if (Config.USE_SECTION):
                self.__model = Model(inputs=[doc_input, aux_input], outputs=preds)
            else:
                self.__model = Model(doc_input, preds)

        #
        # Multi-label classification
        self.__model.compile(loss='binary_crossentropy',
                             # optimizer=optimizers.SGD(lr=0.8, nesterov=True),
                             optimizer=optimizers.Adam(),
                             metrics=['acc'])

        self.__model.summary()


    def loadModel(self, file):
        self.__model = load_model(file, custom_objects={'AttLayer': AttLayer})
        self.__sentEncoder = load_model(os.path.splitext(file)[0]+'_1'+os.path.splitext(file)[1], custom_objects={'AttLayer': AttLayer})

    def saveModel(self, file):
        self.__model.save(file)
        self.__sentEncoder.save(os.path.splitext(file)[0]+'_1'+os.path.splitext(file)[1])

    def run(self, dataset):
        # some logging configurations
        fname = 'rad_log'
        history = History()
        csv_logger = CSVLogger('./{0}_{1}.log'.format(fname, Config.REG_PARAM), separator=',', append=True)

        t1 = time.time()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

        if (Config.USE_SECTION):
            X_train = {'main_input': dataset.get_X_train(), 'aux_input': dataset.section_x_train}
        else:
            X_train = dataset.get_X_train()
        history=self.__model.fit(X_train,
                                 dataset.get_Y_train(),
                                 epochs=Config.NN_PARAMS['epochs'][dataset.k], batch_size=Config.NN_PARAMS['batch_size'][dataset.k], shuffle=Config.SHUFFLE,
                                         callbacks=[
                                                    CustomStopper( mode='max',patience=Config.PATIENCE, verbose=1,restore_best_weights=True, dataset=dataset),
                                                    history,
                                                    csv_logger], verbose=2, validation_split=Config.VALIDATION_SPLIT )

        t2 = time.time()


    def predict(self, dataset):
        return self.validate(dataset)

    def validate(self, dataset):
        # test data Prediction and Performance

        if (Config.USE_SECTION):
            X_test = {'main_input': dataset.get_X_test(), 'aux_input': dataset.section_x_test}
        else:
            X_test = dataset.get_X_test()

        preds = self.__model.predict(X_test)
        ids = dataset.X_test[:, 0].tolist()

        return ids, preds

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def getActivationWeights(self, dataset):
        hidden_word_encoding_out = Model(inputs=self.__sentEncoder.input,
                                         outputs=self.__sentEncoder.get_layer('dense_sentence').output)
        word_context_w = self.__sentEncoder.get_layer('word_attention_layer').get_weights()[0]
        word_context_b = self.__sentEncoder.get_layer('word_attention_layer').get_weights()[1]
        word_context_u = self.__sentEncoder.get_layer('word_attention_layer').get_weights()[2]
 
        activations = []
        for i, X_test in enumerate(dataset.get_X_test()):

            encoded_text = X_test
            normalized_text = dataset.han_x_test_doc[i]
            hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)

            eij = np.dot(np.tanh(np.dot(hidden_word_encodings, word_context_w) + word_context_b),
                         np.expand_dims(word_context_u, axis=-1))
            ai = self.softmax(eij)
            ai = ai.astype(float)
            weighted_input = np.multiply(encoded_text, np.squeeze(ai, -1))
            u_wattention = weighted_input

            # generate word, activation pairs
            encoded_text = encoded_text[-len(dataset.han_x_test):]
            encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in encoded_text]
            word_attention = u_wattention[:len(normalized_text)]
            if (np.max(word_attention) > 0.0):
                word_attention = word_attention / np.expand_dims(np.sum(word_attention, -1), -1)
                word_attention = np.array([attention_seq[0:len(sentence)]
                                             for attention_seq, sentence in zip(word_attention, encoded_text)])
            word_activation_maps = []
            for i, text in enumerate(normalized_text):
                word_activation_maps.append(list(zip(text, word_attention[i])))

            activations.append(word_activation_maps)

        return activations


# Define a custom layer for attention.
class AttLayer(Layer):
    def __init__(self, regularizer=None, **kwargs):
        self.regularizer = regularizer
        self.supports_masking = True
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        CONTEXT_DIM = 100
        assert len(input_shape) == 3
        self.W = self.add_weight(name='W', shape=(input_shape[-1], CONTEXT_DIM), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b', shape=(CONTEXT_DIM,), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u', shape=(CONTEXT_DIM,), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.dot(K.tanh(K.dot(x, self.W) + self.b), K.expand_dims(self.u))
        ai = K.softmax(eij, axis=1)
        if mask is not None:
            ai *= K.expand_dims(K.cast(mask, K.floatx()))

        weighted_input = layers.Multiply()([x, ai])
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return None



class CustomStopper(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 dataset=None): # add argument for starting epoch
        super(CustomStopper, self).__init__(monitor,min_delta,patience,verbose,mode,baseline,restore_best_weights)
        self.__dataset=dataset

    def get_X_train(self):
        if (Config.USE_SECTION):
            return {'main_input': self.__dataset.get_X_train(), 'aux_input': self.__dataset.section_x_train}
        else:
            return self.__dataset.get_X_train()

    def get_X_validation(self):
        if (Config.USE_SECTION):
            return {'main_input': self.__dataset.get_X_validation(), 'aux_input': self.__dataset.section_x_validation}
        else:
            return self.__dataset.get_X_validation()

    def get_Y_train(self):
        self._CustomStopper__dataset.get_Y_train()

    def get_Y_test(self):
        return self.__dataset.get_Y_test()

    def get_Y_validation(self):
        return self.__dataset.get_Y_validation()

    def on_train_begin(self, logs=None):
        self.__val_f1=None
        self.__val_recalls = None
        self.__val_precisions = None
        super(CustomStopper, self).on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.get_X_validation())
                                  )).round()
        val_targ = self.__dataset.get_Y_validation()

        self.__val_f1 = f1_score(val_targ, val_predict, average = 'weighted')
        self.__val_recalls = recall_score(val_targ, val_predict, average = 'weighted')
        self.__val_precision = precision_score(val_targ, val_predict, average = 'weighted')

        print (" — val_f1: % f — val_precision: % f — val_recall % f " % ( self.__val_f1,  self.__val_precision,  self.__val_recalls))
        super(CustomStopper, self).on_epoch_end(epoch=epoch,logs=logs)
        print (" - best ", self.best)
        return

    def get_monitor_value(self, logs):
        return self.__val_f1