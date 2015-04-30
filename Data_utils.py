import cPickle as pickle
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn import *


class Data(object):
    ''' Class qui handles les objects pickled avec extract data'''


    def __init__(self,pix,root):
        ''' Parameter d'entre et attribut '''

        self.pix = pix
        self.input_file = root+'/LOLA'+str(pix)+'_GRAIL_Dataset'
        self.data = self.m_load()
        for key1,val1 in self.data.iteritems():
            setattr(self,key1,val1)
            if type(val1) == dict:
                for key2,val2 in val1.iteritems():
                    setattr(self,key2,val2)

    def m_load(self):
        with open(self.input_file, 'rb') as f:
            try:
                data_pickle = pickle.load(f)
            except:
                data_pickle = pd.read_pickle(self.input_file)
        return data_pickle
                    

    def Plot_histogram(self,col,i):
        ''' Plot a histogram with all the data
        input :
        col : Nom de l'attribut a plotter
        i : index du crater a plotter
        output:
        belle figure'''
        c = sns.color_palette('deep',n_colors = getattr(self,col).shape[0])
        plt.scatter(getattr(self,'h_'+col)[:-1],getattr(self,col)[i,:],color = c[i])


    def Training_Ages(self):
        ''' Return X,y for Age estimation of the crater '''
        
        feature = ('Diameter','Long','Lat',)
        feature += tuple(self.feat_lola.keys(),)
        feature += tuple(self.feat_grail.keys(),)
        X = np.hstack([getattr(self,f) for f in feature])
        y =  self.Age.astype('int')


        X = X[(y != 19).any(axis=1)]
        y =  y[(y != 19).any(axis=1)].ravel()
        
        return X,y
        



        
            

        
