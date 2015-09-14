from PIL import Image
import PIL
import numpy as np
from Data_utils import *
from Extract_Array import *

from sys import platform as _platform
import distutils.util
import shutil
import datetime

    
##############################
# Platform

platform = 'laptop'

if _platform == "linux" or _platform == "linux2":
    racine = '/gpfs/users/thorey/Classification/'
elif _platform == "darwin":
    if platform == 'clavius':
        racine = '/Users/clement/Classification/'
    else:
        racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

data_path = os.path.join(root,'Data')
extraction_path = os.path.join(racine,'Data','ImageExtractionLROC')

#Load data
data = Data(64,data_path,'_2')
df = pd.DataFrame(np.hstack((data.Index,data.Lat,data.Long,data.Diameter)),
                  columns = ['Index','Lat','Long','Diameter'])

df_LROC = 0 # Contient les pixel 
df_LROC_Error = []

# compteur
compteur_init = len(df)
compteur =  compteur_init
tracker = open('tracker_feature.txt','wr+')
tracker.write('Debut du feature extraction \n')
tracker.close()

for i,row in df.iterrows():
    C = Crater(str(int(row.Index)),'i')
    try:
        fig = C.plot_LROC()
    except:
        df_LROC_Error.append(row)
        tracker = open('tracker_feature.txt','a')
        tracker.write('Le crater %d a bugger'%(int(row.Index)))
        tracker.write('Il rest encore %d crater'%(compteur))
        tracker.close()
        comtpeur -= 1
        continue
        
    Name = 'C_'+str(row.Index)+'.jpg'
    fig.savefig(os.path.join(extraction_path,Name),
                rasterized=True,
                dpi=100,
                bbox_inches='tight',
                pad_inches=0.0)
    img = Image.open(os.path.join(extraction_path,Name))
    arr = np.array(img.resize((64,64),Image.ANTIALIAS))
    I  = arr.reshape(64*64,3)
    I2 = I.T.flatten()
    if i == 0:
        df_LROC = I2
    else:
        df_LROC = np.vstack((df_LROC,I2))

    tracker = open('tracker_feature.txt','a')
    tracker.write('Il rest encore %d crater'%(compteur))
    tracker.close()
    comtpeur -= 1

data['feat_LROC'] = df_LROC
data['feat_LROC_Error'] = df_LROC_Error
with open(Output+'LOLA'+str(pix)+'_GRAIL_Dataset_3', 'wb') as fi:
    pickle.dump(data, fi, pickle.HIGHEST_PROTOCOL)
