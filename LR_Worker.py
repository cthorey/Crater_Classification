import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from random import randrange

from sklearn.linear_model import LogisticRegression
from numpy.random import uniform
from sklearn import metrics
from sklearn.cross_validation import KFold
import cPickle
from datetime import datetime
import pymc as pm
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.filters import sobel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from numpy.random import randint

from Data_utils import *
from sklearn.cross_validation import train_test_split
from numpy.random import uniform,randint
from datetime import datetime
import pickle

# Which platform

Iteration = 1e6
platform = 'clavius'

if platform == 'clavius':
    racine = '/Users/clement/Classification/'
elif platform == 'laptop':
    racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

# Repertory to load the model
rep_bm_lr = os.path.join(racine,'Notebook','Best_Model','LR')

# Fichier de base !
dat = os.path.join(racine,'Data')
df = Data(64,dat,'_3')



def get_data(df, val_portion = 0.25):
    # Load the raw training data
    X_trainDF,X_testDF, y_trainDF, y_testDF = df.load_train_test()

    print 'La taille de X_train est de (%d,%d) elements'%(X_trainDF.shape)
    print 'La taille de y_train est de (%d) elements'%(y_trainDF.shape)
    print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_trainDF.ravel() == 1)))
    print 'La proportion de FFC dans le train est de %f'%(float(np.sum(y_trainDF.ravel() == 1))/float(len(y_testDF)))
    print('---'*20)
    print 'La taille de X_test est de (%d,%d) elements'%(X_testDF.shape)
    print 'La taille de y_test est de (%d) elements'%(y_testDF.shape)
    print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_testDF.ravel() == 1)))
    print 'Le proportion de FFC dans le test est de %f'%(float(np.sum(y_testDF.ravel() == 1))/float(len(y_testDF)))
    
    X_trainDF.index = range(len(X_trainDF))
    y_trainDF.index = range(len(y_trainDF))
    X_testDF.index = range(len(X_testDF))
    y_testDF.index = range(len(y_testDF))
    
    return X_trainDF, y_trainDF, X_testDF, y_testDF

X_trainDF, y_trainDF, X_testDF, y_testDF = get_data(df, 0.25)



def Score(x):
    if x>0.5:
        return 1
    else:
        return 0
fa = np.vectorize(Score)

def calc_stats(model,X,y,verbose = 0):
    ypred = fa(model.predict_proba(X)[:,1])
    rocauc = metrics.roc_auc_score(y, ypred)
    recall = metrics.recall_score(y,ypred)
    precision = metrics.precision_score(y,ypred)
    prec, rec, thresholds = metrics.precision_recall_curve(y, ypred)
    aucprecrecal = metrics.auc(prec,rec,reorder=True)
    f1score = metrics.f1_score(y,ypred)
    
    if verbose == 1:
        print '-----'*20
        print 'Le roc auc est de %f'%(rocauc)
        print 'Le recall est de %f'%(recall)
        print 'La precision est de %f'%(precision)
        print 'L auc de precision/recall est de  %f'%(aucprecrecal)
        print 'Le f1-score est de  %f'%(aucprecrecal)
        print '-----'*20
    
    return rocauc,recall,precision,aucprecrecal,f1score


def Compute_Statistic(model,Xtr,ytr,Xval,yval,verbose =0):

    if verbose == 1:
        print '-----'*20 
        print 'Training set'
    roc_train,recal_train,prec_train,auc2_train,f1_train = calc_stats(model,Xtr,ytr,verbose)
    if verbose ==1:
        print '-----'*20 
        print 'Validation set'
    roc_val,recal_val,prec_val,auc2_val,f1_val = calc_stats(model,Xval,yval,verbose)    
    return {'roc-train':roc_train,
            'roc-val':roc_val,
            'f1-train':f1_train,
            'f1-val':f1_val, 
            'recal-train':recal_train,
            'recal-val':recal_val,
            'preci-train':prec_train,
            'preci-val':prec_val}


class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class ExtractImageRavel(BaseEstimator,TransformerMixin):
    
    def fit(self, x , y = None):
        return self
    
    def transform(self, X):
        arr = np.array(X).astype('float')
        return arr
    
class ExtractRawImage(BaseEstimator,TransformerMixin):
    
    def fit(self, x , y = None):
        return self
    
    def transform(self, X):
        arr = np.array(X).astype('float')
        X = arr.reshape(arr.shape[0], 3, 64, 64).transpose(0,2,3,1).astype("float")
        return X
    
class HOGgradient(BaseEstimator,TransformerMixin):
    
    def rgb2gray(self,rgb):
        '''Convert RGB image to grayscale'''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    def fit(self, X , y = None):
        return self
    
    def transform(self,X):
        imgs = []
        for x in X:
            if x.ndim == 3:
                x =self.rgb2gray(x)
            imgs.append(hog(x, orientations = 10).ravel())
        return np.vstack(imgs)
    
class SobelFeature(BaseEstimator,TransformerMixin):
    
    def rgb2gray(self,rgb):
        '''Convert RGB image to grayscale'''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    def fit(self, X , y = None):
        return self

    def transform(self,X):
        imgs = []
        for x in X:
            if x.ndim == 3:
                x =self.rgb2gray(x)
            imgs.append(sobel(x).ravel())
        return np.vstack(imgs)

LROC_col = [f for f in X_trainDF.columns if f.split('_')[0]== 'LROC']

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

Feature = FeatureUnion(transformer_list=[
        ('RawImage',Pipeline([
            ('df', ItemSelector(LROC_col)),
            ('ExtractImageRavel', ExtractImageRavel()),
            ('PCA', PCA(n_components = 10)),
            ('StandardScaler', StandardScaler())
            ])),
        ('HOGFeature',Pipeline([
            ('df', ItemSelector(LROC_col)),
            ('ExtracRawImage', ExtractRawImage()),
            ('HOG', HOGgradient()),
            ('PCA', PCA(n_components = 10)),
            ('StandardScaler', StandardScaler())
                ])),
        ('SobelFeature',Pipeline([
            ('df', ItemSelector(LROC_col)),
            ('ExtracRawImage', ExtractRawImage()),
            ('Sobel', SobelFeature()),
            ('PCA', PCA(n_components = 10)),
            ('StandardScaler', StandardScaler())
                ])),
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
                    ]))],
                       transformer_weights = {'RawImage':0,
                                              'HOGFeature':1,
                                              'SobelFeature':1,
                                              'Rad_Stat':0,
                                              'Floor_Stat':1,
                                              'Rim_Stat':0,
                                              'Theta_Stat':0,
                                              'Lambdahh_Stat':0,
                                              'Lambdah1_Stat':0,
                                              'Lambdah2_Stat':0}
        )


# Debut du run

compteur = 0
proba = pm.Bernoulli('p',0.5)
best_val = 0

kf = KFold(len(X_trainDF),5,shuffle=True,random_state=55)

while compteur < Iteration:
    print compteur
    C = 10**(uniform(-6,-2))
    p = uniform(3,6)
    npca = randrange(5, 30)
    
    which_feature = {k:int(proba.random()) for k in Feature.transformer_weights.keys()}
    which_feature['HOGFeature'] = 1
    which_feature['SobelFeature'] = 1
    Feature.transformer_weights = which_feature
    param = {'SobelFeature__PCA__n_components':npca,
             'RawImage__PCA__n_components':npca,
             'HOGFeature__PCA__n_components':npca}
    Feature.set_params(**param)
    
    scores = []; rocauctr = []; rocaucval = []
    print 'Debut cross-validation'
    for train_index, val_index in kf:
        X_trDF, X_valDF = X_trainDF.iloc[train_index], X_trainDF.iloc[val_index]
        y_trDF, y_valDF = y_trainDF.iloc[train_index], y_trainDF.iloc[val_index]
        
        X_tr = Feature.fit_transform(X_trDF)
        y_tr = np.array(y_trDF)[:,np.newaxis]
        
        X_val = Feature.transform(X_valDF)
        y_val = np.array(y_valDF)[:,np.newaxis]
    
        model = LogisticRegression(penalty='l2',C = C, 
                                 class_weight = {0:1,1:p})
        model.fit(X_tr,y_tr.ravel())
        stats = Compute_Statistic(model,X_tr,y_tr,X_val,y_val,verbose = 0)
        scores.append(stats)
        rocauctr.append(stats['roc-train'])
        rocaucval.append(stats['roc-val'])
    rocval = np.array(rocaucval).mean()
    print 'le Roc auc est de %f'%(rocval)

    if rocval>= 0.8:
        X_train = Feature.fit_transform(X_trainDF)
        y_train = np.array(y_trainDF)[:,np.newaxis]
        model.fit(X_train,y_train.ravel())
        
        name_model = 'LR_roctra-%1.3f_rocval-%1.3f'%(np.array(rocauctr).mean(),np.array(rocaucval).mean())
        name = os.path.join(rep_bm_lr,name_model)
        model_pickle = {'Time': str(datetime.now()),
                        'Feature': Feature,
                        'model': model,
                        'Param': (C,p,which_feature),
                        'Score_Crossval' : scores,
                        'ROCauctr': rocauctr,
                        'ROCaucval': rocaucval}
        with open(name, 'wb') as fi:
            pickle.dump(model_pickle, fi, pickle.HIGHEST_PROTOCOL)
    compteur+=1
