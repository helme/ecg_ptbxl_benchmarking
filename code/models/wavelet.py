from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from models.base_model import ClassificationModel
import pickle
import numpy as np
from tqdm import tqdm
import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
import pywt
import scipy.stats
import multiprocessing
import datetime as dt
from collections import defaultdict, Counter
from keras.layers import Dropout, Dense, Input
from keras.models import Model, Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]
 
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def get_single_ecg_features(signal, waveletname='db6'):
    features = []
    for channel in signal.T:
        list_coeff = pywt.wavedec(channel, wavelet=waveletname, level=5)
        channel_features = []
        for coeff in list_coeff:
            channel_features += get_features(coeff)
        features.append(channel_features)
    return np.array(features).flatten()

def get_ecg_features(ecg_data, parallel=True):
    if parallel:
        pool = multiprocessing.Pool(18)
        return np.array(pool.map(get_single_ecg_features, ecg_data))
    else:
        list_features = []
        for signal in tqdm(ecg_data):
            features = get_single_ecg_features(signal)
            list_features.append(features)
        return np.array(list_features)

# for keras models
#def keras_macro_auroc(y_true, y_pred):
#    return tf.py_func(macro_auroc, (y_true, y_pred), tf.double)

class WaveletModel(ClassificationModel):
    def __init__(self, name, n_classes,  freq, outputfolder, input_shape, regularizer_C=.001, classifier='RF'):
        # Disclaimer: This model assumes equal shapes across all samples!
        # standard parameters
        self.name = name
        self.outputfolder = outputfolder
        self.n_classes = n_classes
        self.freq = freq
        self.regularizer_C = regularizer_C
        self.classifier = classifier
        self.dropout = .25
        self.activation = 'relu'
        self.final_activation = 'sigmoid'
        self.n_dense_dim = 128
        self.epochs=30

    def fit(self, X_train, y_train, X_val, y_val):
        XF_train = get_ecg_features(X_train)
        XF_val = get_ecg_features(X_val)
        
        if self.classifier == 'LR':
            if self.n_classes > 1:
                clf = OneVsRestClassifier(LogisticRegression(C=self.regularizer_C, solver='lbfgs', max_iter=1000, n_jobs=-1))
            else:
                clf = LogisticRegression(C=self.regularizer_C, solver='lbfgs', max_iter=1000, n_jobs=-1)
            clf.fit(XF_train, y_train)
            pickle.dump(clf, open(self.outputfolder+'clf.pkl', 'wb'))
        elif self.classifier == 'RF':
            clf = RandomForestClassifier(n_estimators=1000, n_jobs=16)
            clf.fit(XF_train, y_train)
            pickle.dump(clf, open(self.outputfolder+'clf.pkl', 'wb'))
        elif self.classifier == 'NN':
            # standardize input data
            ss = StandardScaler()
            XFT_train = ss.fit_transform(XF_train)
            XFT_val = ss.transform(XF_val)
            pickle.dump(ss, open(self.outputfolder+'ss.pkl', 'wb'))
            # classification stage
            input_x = Input(shape=(XFT_train.shape[1],))
            x = Dense(self.n_dense_dim, activation=self.activation)(input_x)
            x = Dropout(self.dropout)(x)
            y = Dense(self.n_classes, activation=self.final_activation)(x)
            self.model = Model(input_x, y)
            
            self.model.compile(optimizer='adamax', loss='binary_crossentropy')#, metrics=[keras_macro_auroc])
            # monitor validation error
            mc_loss = ModelCheckpoint(self.outputfolder +'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            #mc_score = ModelCheckpoint(self.outputfolder +'best_score_model.h5', monitor='val_keras_macro_auroc', mode='max', verbose=1, save_best_only=True)
            self.model.fit(XFT_train, y_train, validation_data=(XFT_val, y_val), epochs=self.epochs, batch_size=128, callbacks=[mc_loss])#, mc_score])
            self.model.save(self.outputfolder +'last_model.h5')

    def predict(self, X):
        XF = get_ecg_features(X)
        if self.classifier == 'LR':
            clf = pickle.load(open(self.outputfolder+'clf.pkl', 'rb'))
            if self.n_classes > 1:
                return clf.predict_proba(XF)
            else:
                return clf.predict_proba(XF)[:,1][:,np.newaxis]
        elif self.classifier == 'RF':
            clf = pickle.load(open(self.outputfolder+'clf.pkl', 'rb'))
            y_pred = clf.predict_proba(XF)
            if self.n_classes > 1:
                return np.array([yi[:,1] for yi in y_pred]).T
            else:
                return y_pred[:,1][:,np.newaxis]
        elif self.classifier == 'NN':
            ss = pickle.load(open(self.outputfolder+'ss.pkl', 'rb'))#
            XFT = ss.transform(XF)
            model = load_model(self.outputfolder+'best_loss_model.h5')#'best_score_model.h5', custom_objects={'keras_macro_auroc': keras_macro_auroc})
            return model.predict(XFT)
