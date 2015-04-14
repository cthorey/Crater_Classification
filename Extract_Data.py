###################
# Objectif
# 1/ On veut un tupple qui contient a chaque fois
# le nom du crater, son diametre .... + N* matrice de H + P* matrice de gravi
# avec N etant le nombre de carte lola et P le nombre de carte gravi

###################
# Importation des modules
import numpy as np
import pandas as pd
import pickle
import sys
import os
from Extract_Array import *
import scipy.stats as st

from sys import platform as _platform
import distutils.util
import shutil
import datetime

####################
# Fonction a utiliser
def _unload_pickle(input_file):
    with open(input_file, 'rb') as f:
        df = pd.read_pickle(input_file)
    return df

def Statistic(X):
    return {'mean':np.mean(X),'median':np.median(X),'std':np.std(X),'var':np.var(X),'max':np.max(X),'min':np.min(X)}
##############################
# Platform
platform = 'laptop'
    
if _platform == "linux" or _platform == "linux2":
    Root = '/gpfs/users/thorey/FFC/Classification'
elif _platform == "darwin":
    if platform == 'clavius':
        Root = '/Users/clement/Classification/'
    else:
        Root = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

Path_lola = Root+'PDS_FILE/Lola/' # Path carte LOla
Path_grail = Root+'PDS_FILE/Grail/' 
Output = Root+'Data/'
    
###################
# Carte disponible

img = '16'
List_ldem = [f for f in os.listdir(Path_lola) if f.split('_')[0] == 'ldem'] # liste ldem
List_img = [f for f in List_ldem if f.split('_')[1].split('.')[0] == img and f.split('.')[-1] == 'img'] #liste img files
List_lbl = [f for f in List_ldem if f.split('_')[1].split('.')[0] == img and f.split('.')[-1] == 'lbl'] # liste lbl files
carte_lola = [Path_lola + f.split('.')[0] for f in List_img]

###################
# Carte disponible gravi

carte_grail = [Path_grail + f.split('.')[0] for f in os.listdir(Path_grail) if f.split('.')[-1] == 'dat']

###################
# Debut des operation

# on recupere le dataframe avec tous les craters
Racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/Data/'
Source = 'CRATER_MOON_DATA'
df = _unload_pickle( Racine + Source )
df.index = np.arange( len( df ) )
df = df[ ( df.Diameter > 15 ) & ( df.Diameter < 180 ) ]
# On restreint a certain crater

df = df[df.Name.isin(['Taruntius','Vitello'])]
###################
# On initialize les object LolaImage et GrailImage
MapLolas = [BinaryLolaTable(f) for f in carte_lola] # List object lola
MapGrails = [BinaryGrailTable(f) for f in carte_grail] # List object grail

# MapGrails = []

#Header pour chaque collone a quoi elle correspond !
Triplet = ['Int','Couronne','Ext']
array = tuple(df.columns.tolist(),)+('Lola_Int','Lola_Couronne','Lola_Ext',)
for grail_map in MapGrails:
    array+= tuple([(grail_map.name.split('/')[-1]).split('_')[6]+'_'+f for f in Triplet])
# data.append(array)

compteur_init = len(df)*len(MapLolas)
compteur =  compteur_init
tracker = open('tracker.txt','wr+')

# On lance la boucle
data = None
header = []
for MapLola in MapLolas:
    border = MapLola.Boundary()
    dfmap  = df[(df.Long>border[0]) & (df.Long<border[1]) &(df.Lat>border[2]) & (df.Lat<border[3])]
    ind_border = []
    compt = 0
    
    for i,row in df.iterrows():
        if compteur%100 == 0:
            tracker.write('Il reste encore %d/%d iterations \n'%(compteur,compteur_init))

        Window_Coord = MapLola.Cylindrical_Window(1.3*row.Diameter/2.0,row.Lat,row.Long)
        if (Window_Coord[0]<border[0]) or (Window_Coord[1]>border[1])\
           or (Window_Coord[2]<border[2]) or (Window_Coord[3]>border[3]) or (np.isnan(Window_Coord).sum() !=0 ):
            ind_border.append(i)
        else:
            tmp, Z_Int , tmp = MapLola.Circular_Mask(0.98*row.Diameter/2.0,0.98*row.Diameter/2.0,row.Lat,row.Long)
            tmp , tmp, Z_Ext = MapLola.Circular_Mask(1.1*row.Diameter/2.0,1.3*row.Diameter/2.0,row.Lat,row.Long)
            Z_Couronne = MapLola.Couronne_Mask(0.98*row.Diameter/2.0,1.05*row.Diameter/2.0,row.Lat,row.Long)
            # Profondeur par rapport a la preimpact surace
            Floor = Z_Ext.mean()-Z_Int
            depth_floor,bin_floor = np.histogram(Floor , range = (0, 3000), bins = 300)
            depth_floor /= float(depth_floor.sum())    
            statistic_floor = Statistic(Floor)
                
            # hauteur des rim par rapport a la preimpact surface
            Rim = Z_Couronne-Z_Ext.mean()
            height_rim,bin_rim = np.histogram(Rim, range = (-500, 2000), bins = 300)
            height_rim /= float(height_rim.sum())
            statistic_rim = Statistic(Rim)

            # On remplie le header a la premiere iteration
            if compteur == compteur_init:
                df_col = [f for f in dfmap.columns if f not in ['Name']]
                stat_floor = ['floor'+f for f in statistic_floor.keys()]
                bin_floor = ['Floor_%2.2f'%f for f in bin_floor][:-1]
                stat_rim = ['rim'+f for f in statistic_rim.keys()]
                bin_rim = ['Rim_%2.2f'%f for f in bin_rim][:-1]                
                header = (df_col,bin_floor,stat_floor,bin_rim,stat_rim,)
                
            array =  tuple(np.array([f for f in row.tolist() if not isinstance(f,str)]),)+(depth_floor,statistic_floor.values(),height_rim,statistic_rim.values(),)

            for MapGrail in MapGrails:
                tmp, Z_Int , tmp = MapLola.Circular_Mask(0.98*row.Diameter/2.0,0.98*row.Diameter/2.0,row.Lat,row.Long)
                tmp , tmp , Z_Ext = MapLola.Circular_Mask(1.1*row.Diameter/2.0,1.5*row.Diameter/2.0,row.Lat,row.Long)
                center = Z_Int-Z_Ext.mean()
                anomaly_center,bin_center = np.histogram(center,range = (-80,80),bins = 80)
                anomaly_center /= float(anomaly_center.sum())
                statistic_center = Statistic(center)
                array += (anomaly_center,statistic_center.values(),)
                
                if compteur == compteur_init:
                    composante = (MapGrail.name.split('/')[-1]).split('_')[6]
                    bin_center = [composante+'_%2.2f'%f for f in bin_center][:-1]
                    stat_center = [composante+'_'+f for f in statistic_center.keys()]
                    header += (bin_center,stat_center,)
            
            # On range tout dans un gros tableau
            if data == None:
                data = np.hstack((array))
            else:
                data = np.vstack((data,np.hstack(array)))

            compteur-=1
pickle_object = pd.DataFrame(data , columns = np.hstack(header))
with open(Output+'LOLA'+img+'_GRAIL_Dataset', 'wb') as fi:
    pickle.dump(pickle_object, fi, pickle.HIGHEST_PROTOCOL)



