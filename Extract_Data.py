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
            
##############################
# Platform
platform = 'clavius'
    
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

img = '64'
List_ldem = [f for f in os.listdir(Path_lola) if f.split('_')[0] == 'ldem'] # liste ldem
List_img = [f for f in List_ldem if f.split('_')[1].split('.')[0] == img and f.split('.')[-1] == 'img'] #liste img files
List_lbl = [f for f in List_ldem if f.split('_')[1].split('.')[0] == img and f.split('.')[-1] == 'lbl'] # liste lbl files
carte_lola = [Path_lola + f.split('.')[0] for f in List_img]

###################
# Carte disponible gravi

carte_grail = [Path_grail + f.split(' . ')[0] for f in os.listdir(Path_grail) if f.split(' . ')[-1] == 'dat']

###################
# Debut des operation

# on recupere le dataframe avec tous les craters
Racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/Data/'
Source = 'CRATER_MOON_DATA'
df = _unload_pickle( Racine + Source )
df.index = np.arange( len( df ) )
df = df[ ( np.abs(df.Lat) < 75 ) & ( df.Diameter > 20 ) & ( df.Diameter < 180 ) ]
# On restreint a certain crater

###################
# On initialize les object LolaImage et GrailImage
MapLolas = [BinaryLolaTable(f) for f in carte_lola] # List object lola
MapGrails = [BinaryGrailTable(f) for f in carte_grail] # List object grail

#Header pour la premiere ligne
data = []
Triplet = ['Int','Couronne','Ext']
array = tuple(df.columns.tolist(),)+('Lola_Int','Lola_Couronne','Lola_Ext',)
for grail_map in MapGrails:
    array+= tuple([(grail_map.name.split('/')[-1]).split('_')[6]+'_'+f for f in Triplet])
data.append(array)

compteur_init = len(df)*len(MapLolas)
compteur =  compteur_init

# On lance la boucle
for MapLola in MapLolas:
    border = MapLola.Boundary()
    dfmap  = df[(df.Long>border[0]) & (df.Long<border[1]) &(df.Lat>border[2]) & (df.Lat<border[3])]
    ind_border = []
    compt = 0
    for i,row in df.iterrows():
        compteur-=1
        print 'Il reste encore %d/%d iterations '%(compteur,compteur_init)
        D = (1.5*row.Diameter/2.0)*360/(2*np.pi*1734.4) # Diameter en degres
        if (row.Long-D<border[0]) or (row.Long+D>border[1]) or (row.Lat-D<border[2]) or (row.Lat+D>border[3]):
            ind_border.append(i)
        else:
            Z_Int , tmp = MapLola.Circular_Mask(0.98*row.Diameter/2.0,0.98*row.Diameter/2.0,row.Lat,row.Long)
            tmp , Z_Ext = MapLola.Circular_Mask(1.05*row.Diameter/2.0,1.5*row.Diameter/2.0,row.Lat,row.Long)
            Z_Couronne,tmp = MapLola.Couronne_Mask(0.98*row.Diameter/2.0,1.05*row.Diameter/2.0,row.Lat,row.Long)
            array = tuple(row.tolist(),)+(Z_Int,Z_Couronne,Z_Ext,)
            for MapGrail in MapGrails:
                Z_Int , tmp = MapGrail.Circular_Mask(0.98*row.Diameter/2.0,0.98*row.Diameter/2.0,row.Lat,row.Long)
                tmp , Z_Ext = MapGrail.Circular_Mask(1.05*row.Diameter/2.0,1.5*row.Diameter/2.0,row.Lat,row.Long)
                Z_Couronne,tmp = MapGrail.Couronne_Mask(0.98*row.Diameter/2.0,1.05*row.Diameter/2.0,row.Lat,row.Long)
                array += (Z_Int,Z_Couronne,Z_Ext,)
            data.append(array)
        compt+=1

with open(Output+'LOLA'+img+'_GRAIL_Dataset', 'wb') as fi:
    pickle.dump(data, fi, pickle.HIGHEST_PROTOCOL)



