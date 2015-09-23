import cPickle as pickle
import numpy as np
import seaborn as sns
import os
import matplotlib.pylab as plt
import pandas as pd
from sklearn.cross_validation import train_test_split


class Data(object):
    ''' Class qui handles les objects pickled avec extract data'''

    def __init__(self,pix,root,version):
        ''' Parameter d'entre et attribut '''
        self.version = version
        self.racine = root
        self.pix = pix
        self.input_file = root+'/LOLA'+str(pix)+'_GRAIL_Dataset'+version
        self.data = self.m_load()
        for key1,val1 in self.data.iteritems():
           setattr(self,key1,val1)
           if type(val1) == dict:
               for key2,val2 in val1.iteritems():
                   setattr(self,key2,val2)
        if version == '_3':
            self.Load_Image_Data()

    def m_load(self):
        with open(self.input_file, 'rb') as f:
            try:
                data_pickle = pickle.load(f)
            except:
                data_pickle = pd.read_pickle(self.input_file)
        return data_pickle
                    

    def Load_Image_Data(self):

        if len(self.data['feat_LROC_Error'])>0:
            print 'Attention, il y a des erreur ds ce dataset'
            print 'Les indices ne vont pas matcher'
            
        X = self.data['feat_LROC']
        self.LROC = X.reshape(X.shape[0], 3, 64, 64).transpose(0,2,3,1).astype("float")
        
    def Plot_histogram(self,col,i):
        ''' Plot a histogram with all the data
        input :
        col : Nom de l'attribut a plotter
        i : index du crater a plotter
        output:
        belle figure'''
        c = sns.color_palette('deep',n_colors = getattr(self,col).shape[0])
        plt.scatter(getattr(self,'h_'+col)[:-1],getattr(self,col)[i,:],color = c[i])


    def Ages_Array(self):
        ''' Return X,y for Age estimation of the crater '''
        
        feature = ('Diameter','Long','Lat',)
        feature += tuple(self.feat_lola.keys(),)
        feature += tuple(self.feat_grail.keys(),)
        X = np.hstack([getattr(self,f) for f in feature])
        y =  self.Age.astype('int')


        X = X[(y != 19).any(axis=1)]
        y =  y[(y != 19).any(axis=1)].ravel()
        
        return X,y

    def Ages_data(self):
        ''' Return lequivalent de data la ou on connait l age'''

        data_tmp  = {}
        keys = [f for f in self.data.keys() if not f.split('_')[0] in ['h','failed']] # Enleve les cat inutiles
        indice = np.where(self.Age.astype('int')<19)[0] # Indice des element dont on connait l'age'
        indice = np.where((self.Age.astype('int')<19)&(self.Type.astype('int') ==0))[0]
        for key in keys:
            data_tmp[key] = {k:v[indice] for k,v in self.data[key].iteritems()}
            
        return data_tmp
        
    def str2bool(self,elt):
        if elt  == 'False':
            return False
        else:
            return True
            
    def rawDF(self):
        ''' Prend un objet Data en imput e
        renvoie un DataFrame avec toutes les composantes 
        possible '''
    
        train = pd.DataFrame(np.hstack(self.data['feat_df'].values()),
                             columns = self.data['feat_df'].keys()).convert_objects(convert_numeric = True)
        for key,val in self.data['feat_lola'].iteritems():
            if 'stat' not in key.split('_'):
                train = train.join(
                    pd.DataFrame(val , columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                 self.data['h_feat_lola']['h_'+key])[:-1]))
            else:
                train = train.join(
                    pd.DataFrame(val ,columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                self.data['h_feat_lola']['h_'+key])))
    
        for key,val in self.data['feat_grail'].iteritems():
            if 'stat' not in key.split('_'):
                train = train.join(
                    pd.DataFrame(val , columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                 self.data['h_feat_grail']['h_'+key])[:-1]))
            else:
                train = train.join(
                    pd.DataFrame(val ,columns = map(lambda x: 'h_'+key+'_'+str(x),
                                                self.data['h_feat_grail']['h_'+key])))
                
        mask = [f for f in train.columns if f.split('_')[0]== 'Mask']
        train[mask] = train[mask].applymap(self.str2bool)
        
        if self.version == '_3':
            col_LROC = ['LROC_'+str(f) for f in range(self.data['feat_LROC'].shape[1])]
            trainLROC = pd.DataFrame(self.data['feat_LROC'] , columns = col_LROC)            
            train = train.join(trainLROC)

        rawdata = train.convert_objects(convert_numeric=True)

        return rawdata

    def TrainingDF(self):

        train = self.rawDF()
        print 'On restreint au crater larger than 20 km'    
        train = train[train.Diameter>=20.0]
        print 'On restreint au crater proche des FFCs'
        train = train[train.Mask_C_FFC]
        
        train.index = range(len(train))
        labels = train.Type
        train = train.drop('Type', 1)
        return train , labels

    def To_DetermineDF(self):

        data = self.rawDF()
        print 'On restreint au crater larger than 20 km'    
        data = data[data.Diameter>=20.0]
        print 'On selectionne tous les crater absent du training'
        data = data[~data.Mask_C_FFC]
        data = data.drop('Type', 1)
        data = data.dropna()
        return data 

    def Generate_Index_Train_Test(self):

        trainDF, labels = self.TrainingDF()
        X_trainDF,X_testDF, y_trainDF, y_testDF = train_test_split(trainDF,labels,test_size=0.3, random_state=42)

        print('---'*20)
        print 'La taille de X_train est de (%d,%d) elements'%(X_trainDF.shape)
        print 'La taille de y_train est de (%d) elements'%(y_trainDF.shape)
        print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_trainDF.ravel() == 1)))
        print 'La proportion de FFC dans le train est de %f'%(float(np.sum(y_trainDF.ravel() == 1))/float(len(y_testDF)))
        print('---'*20)
        print 'La taille de X_test est de (%d,%d) elements'%(X_testDF.shape)
        print 'La taille de y_test est de (%d) elements'%(y_testDF.shape)
        print 'Le nombre de  FFC dans le train est de %d'%(float(np.sum(y_testDF.ravel() == 1)))
        print 'Le proportion de FFC dans le test est de %f'%(float(np.sum(y_testDF.ravel() == 1))/float(len(y_testDF)))

        idxs = {'idxTrain': np.array(X_trainDF.index),
                'idxTest':np.array(X_testDF.index)}
        idxname = os.path.join(self.racine,'idxs')
        with open(idxname, 'wb') as fi:
            pickle.dump(idxs, fi, pickle.HIGHEST_PROTOCOL)

    def load_train_test(self):
        
        trainDF, labels = self.TrainingDF()
        name = os.path.join(self.racine,'idxs')
        with open(name, 'rb') as f:
            idxs = pickle.load(f)

        X_train = trainDF.iloc[idxs['idxTrain']]
        y_train = labels.iloc[idxs['idxTrain']]
        X_test = trainDF.iloc[idxs['idxTest']]
        y_test = labels.iloc[idxs['idxTest']]            

        return X_train,X_test,y_train,y_test
        
            

        
            

        
