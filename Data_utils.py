import cPickle as pickle
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


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
                    

    def Plot_histogram(self,col):
        ''' Plot a histogram with all the data '''
        c = sns.color_palette('deep',n_colors = getattr(self,col).shape[0])
        for i in range(getattr(self,col).shape[0]):
            plt.scatter(getattr(self,'h_'+col)[:-1],getattr(self,col)[i,:],color = c[i])
            

        
