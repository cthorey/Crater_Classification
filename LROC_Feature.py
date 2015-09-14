from PIL import Image
import PIL
import numpy as np
from Data_utils import *
from Extract_Array import *

import pylab
from sys import platform as _platform
import distutils.util
import shutil
import datetime
import matplotlib as mpl 

    
##############################
# Platform

platform = 'clavius'

if _platform == "linux" or _platform == "linux2":
    racine = '/gpfs/users/thorey/Classification/'
elif _platform == "darwin":
    if platform == 'clavius':
        racine = '/Users/clement/Classification/'
    else:
        racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

data_path = os.path.join(racine,'Data')
extraction_path = os.path.join(racine,'Data','ImageExtractionLROC')

#Load data
data = Data(64,data_path,'_2')
df = pd.DataFrame(np.hstack((data.Index,data.Lat,data.Long,data.Diameter)),
                  columns = ['Index','Lat','Long','Diameter'])

df_LROC = 0 # Contient les pixel 
df_LROC_Error = []

# Thrreshold to black

def m_initialize_mpl():
    mpl.rcdefaults()
    mpl.rcParams['figure.facecolor'] = 'w'
    mpl.rcParams['lines.linewidth'] =1.5
def is_White(x):
    if x == 255:
        return 0
    else:
        return x
f = np.vectorize(is_White, otypes=[np.uint8])

m_initialize_mpl()
# compteur
compteur_init = len(df)
compteur =  compteur_init
tracker = open('tracker_feature.txt','wr+')
tracker.write('Debut du feature extraction \n')
tracker.close()

df = df[df.Index == 6]
for i,row in df.iterrows():
    C = Crater(str(int(row.Index)),'i',racine)   
    try:
        fig = C.plot_LROC()
    except:
        df_LROC_Error.append(row)
        tracker = open('tracker_feature.txt','a')
        tracker.write('Le crater %d a bugger \n'%(int(row.Index)))
        tracker.write('Il rest encore %d crater \n'%(compteur))
        tracker.close()
        comtpeur -= 1
        continue
        
    Name = 'C_'+str(int(row.Index))+'.png'
    # On travaille avec ds png sinonb fonction pas
    # sur clavius, juste rajoute une 4 eme depth qui
    # correposnd a alpha que l'on enleve
    
    fig.savefig(os.path.join(extraction_path,Name),
                rasterized=True,
                dpi=100,
                bbox_inches='tight',
                pad_inches=0.0,
                facecolor='w',
                edgecolor='w')
    plt.close()
    sys.exit()
    img = Image.open(os.path.join(extraction_path,Name))
    img = Image.fromarray(f(np.array(img)[:,:,:-1]))
    arr = np.array(img.resize((64,64),Image.ANTIALIAS))
    I  = arr.reshape(64*64,3)
    I2 = I.T.flatten()
    if i == 0:
        df_LROC = I2
    else:
        df_LROC = np.vstack((df_LROC,I2))

    tracker = open('tracker_feature.txt','a')
    tracker.write('Il rest encore %d crater \n'%(compteur))
    tracker.close()
    compteur -= 1

data.data['feat_LROC'] = df_LROC
data.data['feat_LROC_Error'] = df_LROC_Error

Output = os.path.join(data_path,'LOLA'+str(64)+'_GRAIL_Dataset_3')
with open(Output, 'wb') as fi:
    pickle.dump(data, fi, pickle.HIGHEST_PROTOCOL)
