# coding: utf-8

import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from keras.utils import np_utils
from keras import callbacks
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2
from keras import optimizers
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Data_utils import *

from numpy.random import uniform,randint
from datetime import datetime
import pickle

# Which platform

platform = 'laptop'

if platform == 'clavius':
    racine = '/Users/clement/Classification/'
elif platform == 'laptop':
    racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

# Repertory to load the model
rep_bm = os.path.join(racine,'Notebook','Best_Model')
rep_bm_nn = os.path.join(rep_bm,'Neural_Net')

# Fichier de base !
dat = os.path.join(racine,'Data')
df = Data(64,dat,'_2')

def Training_df(df):
    ''' Prend un objet Data en imput e
    renvoie un DataFrame avec toutes les composantes 
    possible '''
    
    train = pd.DataFrame(np.hstack(df.data['feat_df'].values()),
                         columns = df.data['feat_df'].keys()).convert_objects(convert_numeric = True)
    for key,val in df.data['feat_lola'].iteritems():
        if 'stat' not in key.split('_'):
            train = train.join(
                pd.DataFrame(val , columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                 df.data['h_feat_lola']['h_'+key])[:-1]))
        else:
            train = train.join(
                pd.DataFrame(val ,columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                df.data['h_feat_lola']['h_'+key])))
    
    for key,val in df.data['feat_grail'].iteritems():
        if 'stat' not in key.split('_'):
            train = train.join(
                pd.DataFrame(val , columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                 df.data['h_feat_grail']['h_'+key])[:-1]))
        else:
            train = train.join(
                pd.DataFrame(val ,columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                df.data['h_feat_grail']['h_'+key])))
    def str2bool(elt):
        if elt  == 'False':
            return False
        else:
            return True
    mask = [f for f in train.columns if f.split('_')[0]== 'Mask']
    train[mask] = train[mask].applymap(str2bool)
    
    return train.convert_objects(convert_numeric=True)

raw = Training_df(df)
trainDF = raw[raw.Mask_C_FFC]


# On separe le label
X  = trainDF.drop('Type', 1)
y =  trainDF.Type

# On split
X_traintotalDF, X_testDF, y_traintotalDF, y_testDF = train_test_split(X,y,test_size = 0.25)
X_trainDF, X_valDF, y_trainDF, y_valDF = train_test_split(X_traintotalDF,y_traintotalDF,test_size = 0.3)

print('---'*20)
print 'La taille de X_train est de (%d,%d) elements'%(X_trainDF.shape)
print 'La taille de y_train est de (%d) elements'%(y_trainDF.shape)
print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_trainDF.ravel() == 1)))
print 'La proportion de FFC dans le train est de %f'%(float(np.sum(y_trainDF.ravel() == 1))/float(len(y_testDF)))
print('---'*20)
print 'La taille de X_val est de (%d,%d) elements'%(X_valDF.shape)
print 'La taille de y_val est de (%d) elements'%(y_valDF.shape)
print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_valDF.ravel() == 1)))
print 'Le proportion de  FFC dans le val est de %f'%(float(np.sum(y_valDF.ravel() == 1))/float(len(y_valDF)))
print('---'*20)
print 'La taille de X_test est de (%d,%d) elements'%(X_testDF.shape)
print 'La taille de y_test est de (%d) elements'%(y_testDF.shape)
print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_testDF.ravel() == 1)))
print 'Le proportion de FFC dans le test est de %f'%(float(np.sum(y_testDF.ravel() == 1))/float(len(y_testDF)))

    
# Feature extraction

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples """
    
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    
class ColumVector(BaseEstimator,TransformerMixin):
    'Handle le cas ou on retourne une serie'
    
    def fit(self,x, y = None):
        return self
    
    def transform(self, X , **transform_params):
        return X[:,np.newaxis]

# topo
depth_floor_histo = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_depth_floor']
depth_rim_histo = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_height_rim']

stat_floor = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_stat_floor']
stat_rim = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_stat_rim']

stat_rad = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_rad_stat']
val_rad = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_rad_anomaly']

stat_theta = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_theta_stat']
val_theta = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_theta_anomaly']

stat_lambdah1 = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_lambdah1_stat']
val_lambdah1 = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_lambdah1_anomaly']

stat_lambdah2 = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_lambdah2_stat']
val_lambdah2 = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_lambdah2_anomaly']

stat_lambdahh = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_lambdahh_stat']
val_lambdahh = [f for f in X_trainDF.columns.tolist() if '_'.join(f.split('_')[:3]) == 'h_lambdahh_anomaly']

Features_Col = ['Lat','Lon']+ stat_floor + stat_rim + stat_rad + stat_theta + stat_lambdahh
Features = FeatureUnion( transformer_list=[
    ('Floor_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_floor)),
        ('Scaler', StandardScaler()),
    ])),
    ('Rim_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_rim)),
        ('Scaler', StandardScaler()),
    ])),
    ('Rad_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_rad)),
        ('Scaler', StandardScaler()),
    ])),
    ('Theta_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_theta)),
        ('Scaler', StandardScaler()),
    ])),
    ('Lambdahh_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_lambdahh)),
        ('Scaler', StandardScaler())
    ])),
    ('Lambdah1_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_lambdah1)),
        ('Scaler', StandardScaler())
    ])),
    ('Lambdah2_Stat', Pipeline([
        ('Selector', ItemSelector(key= stat_lambdah2)),
        ('Scaler', StandardScaler())
    ]))
])

def shuffle(X, y, seed=1337):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y

# On entrain Feature sur le training set et on applique la meme transformation sur els autres
Features.fit(X_trainDF)
X_train = Features.transform(X_trainDF)
y_train = np.array(y_trainDF)[:,np.newaxis]
Y_train = np_utils.to_categorical(y_train)

X_val = Features.transform(X_valDF)
y_val = np.array(y_valDF)[:,np.newaxis]
Y_val = np_utils.to_categorical(y_val)

X_train_total = Features.transform(X_traintotalDF)
y_train_total = np.array(y_traintotalDF)[:,np.newaxis]
Y_train_total = np_utils.to_categorical(y_train_total)

X_test = Features.transform(X_testDF)
y_test = np.array(y_testDF)[:,np.newaxis]
Y_test = np_utils.to_categorical(y_test)


def Plot_statistic(history):
    fig = plt.figure(figsize=(18,10))
    plt.subplot(131)
    plt.plot(history.losses_train,label = 'train')
    plt.plot(history.losses_val, label = 'val')
    plt.legend()
    plt.title('Loss')
    plt.subplot(132)
    plt.plot(history.roc_train, label = 'train')
    plt.plot(history.roc_val , label = 'val')
    plt.title('ROC AUC Score')
    plt.legend()
    plt.subplot(133)
    plt.plot(np.array(history.metric_train)[:,0] , label = 'train')
    plt.plot(np.array(history.metric_val)[:,0] , label = 'val')
    plt.title('f1 Score')
    plt.legend()
    
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []
        self.roc_train = []
        self.roc_val = []
        self.roc_train = []
        self.f1_train = []
        self.f1_val = []
        self.recal_train = []
        self.recal_val = []
        self.preci_train = []
        self.preci_val = []
        
    def on_epoch_end(self, batch, logs={}):
        # losses
        self.losses_train.append(self.model.evaluate(X_train, Y_train, batch_size=128,verbose =0))
        self.losses_val.append(self.model.evaluate(X_val, Y_val, batch_size=128,verbose = 0))
        
        # Roc train
        train_preds = self.model.predict_proba(X_train, verbose=0)
        train_preds = train_preds[:, 1]
        roc_train = metrics.roc_auc_score(y_train, train_preds)
        self.roc_train.append(roc_train)
        
        # Roc val
        val_preds = self.model.predict_proba(X_val, verbose=0)
        val_preds = val_preds[:, 1]
        roc_val = metrics.roc_auc_score(y_val, val_preds)
        self.roc_val.append(roc_val)
        
        # Metrics train
        y_preds = self.model.predict_classes(X_train,verbose = 0)
        self.f1_train.append(metrics.f1_score(y_train,y_preds))
        self.recal_train.append(metrics.recall_score(y_train,y_preds))
        self.preci_train.append(metrics.precision_score(y_train,y_preds))
        
        # Metrics val
        y_preds = self.model.predict_classes(X_val,verbose =0)
        self.f1_val.append(metrics.f1_score(y_val,y_preds))
        self.recal_val.append(metrics.recall_score(y_val,y_preds))
        self.preci_val.append(metrics.precision_score(y_val,y_preds))

def build_model(input_dim, output_dim,nn,reg,opt):
    model = Sequential()
    model.add(Dense(input_dim, nn, init='glorot_uniform',W_regularizer=l2(reg)))
    model.add(PReLU((nn,)))
    model.add(BatchNormalization((nn,)))
    
    model.add(Dense(nn, nn, init='glorot_uniform',W_regularizer=l2(reg)))
    model.add(PReLU((nn,)))
    model.add(BatchNormalization((nn,)))
    
    model.add(Dense(nn, nn, init='glorot_uniform',W_regularizer=l2(reg)))
    model.add(PReLU((nn,)))
    model.add(BatchNormalization((nn,)))

    model.add(Dense(nn, output_dim, init='glorot_uniform',W_regularizer=l2(reg)))
    model.add(Activation('softmax'))
    
    model.compile(loss='binary_crossentropy', optimizer= opt)
    return model

# Model training

# On entrain Feature sur le training set et on applique la meme transformation sur els autres
Features.fit(X_trainDF)
X_train = Features.transform(X_trainDF)
y_train = np.array(y_trainDF)[:,np.newaxis]
Y_train = np_utils.to_categorical(y_train)

X_val = Features.transform(X_valDF)
y_val = np.array(y_valDF)[:,np.newaxis]
Y_val = np_utils.to_categorical(y_val)

X_train_total = Features.transform(X_traintotalDF)
y_train_total = np.array(y_traintotalDF)[:,np.newaxis]
Y_train_total = np_utils.to_categorical(y_train_total)

X_test = Features.transform(X_testDF)
y_test = np.array(y_testDF)[:,np.newaxis]
Y_test = np_utils.to_categorical(y_test)


input_dim = X_train.shape[1]
output_dim = 2

reg_inf = -4;reg_sup = 0
neuron = [32,64,128,256]
optimizer = ['adam','RMSprop']
worker = 1
while worker != 0:
    reg = 10**(uniform(reg_inf,reg_sup))
    nn = neuron[randint(len(neuron))]
    opt = optimizer[randint(len(optimizer))]
    
    # build model 
    model = build_model(input_dim, output_dim,nn,reg,opt)

    # training model
    history = LossHistory()
    model.fit(X_train, Y_train, 
          nb_epoch=200, 
          batch_size = 32,
          verbose=0,
          callbacks = [history])
    roc_val = history.roc_val[-1]
    f1_sc = history.f1_val[-1]
    print roc_val,f1_sc
    if roc_val>0.8 and f1_sc>0.1 :
        name_model = 'NN_ROC-%1.3f_f1-%1.3f'%(roc_val,f1_sc)
        name = os.path.join(rep_bm_nn,name_model)
        model.save_weights(name+".h5")
        history.model = history.model.to_json()
        model_pickle = {'Time': str(datetime.now()),
                        'model': model.to_json(),
                        'history': history,
                        'roc' : roc_val,
                        'f1': f1_sc}
        with open(name, 'wb') as fi:
            pickle.dump(model_pickle, fi, pickle.HIGHEST_PROTOCOL)
        worker = 0

    
