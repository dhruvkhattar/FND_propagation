import json
import pdb
import numpy as np
import pickle as pkl
from ast import literal_eval
import json
from tqdm import tqdm
import keras
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout, GlobalAveragePooling1D
from keras.layers.merge import concatenate, dot, multiply, add
from keras.models import Model
from keras.callbacks import ModelCheckpoint


class RNN_Model:

    def __init__(self, tweet_path_file, user_features_file, label_file, sequence_len, n_features):

        self.tweet_path_fp = open(tweet_path_file)
        self.user_features_fp = open(user_features_file)
        self.label_fp = open(label_file)
        self.sequence_len = sequence_len
        self.n_features = n_features


    def data_handler(self):
        
        self.user_features = literal_eval(self.user_features_fp.read())
        self.tweet_path = literal_eval(self.tweet_path_fp.read())
        self.labels = literal_eval(self.label_fp.read())
        self.tweets = self.tweet_path.keys()
        self.users = self.user_features.keys()

        self.inputs = []
        self.outputs = []

        for tweet in tqdm(self.tweets):
            if len(self.tweet_path[tweet]) >= self.sequence_len:
                user_list = [i[0] for i in self.tweet_path[tweet][:self.sequence_len]]
                user_list_features = map(lambda x: self.user_features[x], user_list)
                self.inputs.append(user_list_features)
            else:
                user_list = [i[0] for i in self.tweet_path[tweet]]
                user_list_features = map(lambda x: self.user_features[x], user_list)
                user_list_features = 40*user_list_features
                self.inputs.append(user_list_features[:self.sequence_len])
            self.outputs.append(self.labels[tweet])
        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        
    def test_data_handler(self):
        
        self.user_features = literal_eval(self.user_features_fp.read())
        self.tweet_path = literal_eval(self.tweet_path_fp.read())
        self.tweets = self.tweet_path.keys()
        self.users = self.user_features.keys()

        self.tweet_features = {}

        for tweet in tqdm(self.tweets):
            if len(self.tweet_path[tweet]) >= self.sequence_len:
                user_list = [i[0] for i in self.tweet_path[tweet][:self.sequence_len]]
                user_list_features = map(lambda x: self.user_features[x], user_list)
                self.tweet_features[tweet] = user_list_features
            else:
                user_list = [i[0] for i in self.tweet_path[tweet]]
                user_list_features = map(lambda x: self.user_features[x], user_list)
                user_list_features = 40*user_list_features
                self.tweet_features[tweet] = user_list_features[:self.sequence_len]

        
    def create_model(self):

        user_features = Input(shape=(self.sequence_len, self.n_features))
        
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(user_features)
        avg_layer = GlobalAveragePooling1D()(lstm_layer)
        dense1 = Dense(256, activation='relu')(avg_layer)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        output = Dense(1, activation='sigmoid')(dropout2)

        self.model = Model(inputs=[user_features], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        

    def fit_model(self, inputs, outputs, epochs):
        filepath="./weights/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)

    
def train(tweet_file, user_file, label_file, sequence_len, n_features):
    model = RNN_Model(tweet_file, user_file, label_file, sequence_len, n_features)
    model.data_handler()
    model.create_model()
    model.model.summary()
    model.fit_model([model.inputs], model.outputs, 50)


def test(tweet_file, user_file, label_file, sequence_len, n_features):
    print("Loading Model")
    model = RNN_Model(tweet_file, user_file, label_file, sequence_len, n_features)
    model.test_data_handler()
    model.create_model()
    model.model.load_weights('./weights.hdf5')
    #tweet_ids = pkl.load(open('./Twitter18/tweet-id.pkl'))
    print("RNN Model loaded")
    while True:
        tweet_link = raw_input('Enter the tweet URL: ')
        #tweet = tweet.decode('utf-8')
        #pdb.set_trace()
        #features = np.array(model.tweet_features[str(tweet_ids[tweet.strip()])])
        features = np.array(model.tweet_features[tweet_link.split('/')[-1]])
        features = np.expand_dims(features, 0)
        print(model.model.predict(features)[0][0])


if  __name__ == '__main__':

    #train('./Twitter18/retweet_path.txt', './Twitter18/user_feature.txt', './Twitter18/label.txt', 40, 20)
    test('./Twitter18/retweet_path.txt', './Twitter18/user_feature.txt', './Twitter18/label.txt', 40, 20)
