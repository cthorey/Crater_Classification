import cPickle as pickle
import numpy as np

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
                    
