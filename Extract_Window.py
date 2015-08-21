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

    
##############################
# Platform
platform = 'laptop'
pix = '64'
    
if _platform == "linux" or _platform == "linux2":
    Root = '/gpfs/users/thorey/Classification/'
elif _platform == "darwin":
    if platform == 'clavius':
        Root = '/Users/clement/Classification/'
    else:
        Root = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

Path_lola = Root+'PDS_FILE/Lola/' # Path carte LOla
Path_grail = Root+'PDS_FILE/Grail/' 
Output = Root+'Data/'

####################
# Fonction a utiliser
def _unload_pickle(input_file):
    with open(input_file, 'rb') as f:
        df = pd.read_pickle(input_file)
    return df

def Carte_Lola(Path_lola,img):
    """ Instancie des object BinaryLolaTable
    input:
        Path_lola : path ou sont range les fichier .img .lbl
        img : Nombre de ppd (pixel par degree)
    output:
       MapLolas : liste des objects
    """
    
    List_ldem = [f for f in os.listdir(Path_lola) if f.split('_')[0] == 'ldem'] # liste ldem
    List_img = [f for f in List_ldem if f.split('_')[1].split('.')[0] == img and f.split('.')[-1] == 'img'] #liste img files
    List_lbl = [f for f in List_ldem if f.split('_')[1].split('.')[0] == img and f.split('.')[-1] == 'lbl'] # liste lbl files
    carte_lola = [Path_lola + f.split('.')[0] for f in List_img]
    return carte_lola

def Carte_Grail(Path_grail):
    """ Instancie des object BinaryGrailTable
    input:
    Path_grail : path ou sont range les fichier .dat
    output:
    MapGrails : liste des objects
    """
    carte_grail = [Path_grail + f.split('.')[0] for f in os.listdir(Path_grail) if f.split('.')[-1] == 'dat']
    MapGrails = [BinaryGrailTable(f) for f in carte_grail] # List object grail
    return MapGrails
    
def Extract_Array_Lola(MapLola,row,lola_tmp):
    """ Retourne des 1D tables de l'interieur du crater, l'exterieur et le rim
    Input
        - MapLola : Object class LolaBinaryTable
        - row : row Dataframe
    Output :
        - '(h_ZI,h_ZE,h_ZR)'
        - '(Z_Int,Z_Ext,Z_Rim)'
    '"""
    tmp, Z_Int , tmp = MapLola.Circular_Mask(0.98*row.Diameter/2.0,0.98*row.Diameter/2.0,row.Lat,row.Long)
    tmp , tmp, Z_Ext = MapLola.Circular_Mask(1.1*row.Diameter/2.0,1.3*row.Diameter/2.0,row.Lat,row.Long)
    Z_Rim = MapLola.Couronne_Mask(0.98*row.Diameter/2.0,1.05*row.Diameter/2.0,row.Lat,row.Long)

    lola_tmp['L_ZInt'] = Z_Int.astype('int16')
    lola_tmp['L_ZExt'] = Z_Ext.astype('int16')
    lola_tmp['L_ZRim'] = Z_Rim.astype('int16')
    
    return lola_tmp
        
def Extract_Array_Grail(MapGrail,row,grail_tmp,key):

    """ Retourne des 1D tables de l'interieur du crater, l'exterieur et le rim
    Input
    - MapGrail : Object class LolaBinaryTable
    - row : row Dataframe
    Output :
    - '(h_ZI,h_ZE)'
    - '(Z_Int,Z_Ext)'
    '"""
    tmp, Z_Int , tmp = MapGrail.Circular_Mask(0.98*row.Diameter/2.0,0.98*row.Diameter/2.0,row.Lat,row.Long)
    tmp , tmp , Z_Ext = MapGrail.Circular_Mask(1.1*row.Diameter/2.0,1.5*row.Diameter/2.0,row.Lat,row.Long)

    grail_tmp[key+'_ZInt'] = Z_Int
    grail_tmp[key+'_ZExt'] =Z_Ext
    return grail_tmp

def _Class_Type(x):
    if np.isnan(x):
        return 0
    else:
        return x
        
def Crater_Data(Source):
    Source = Source+'CRATER_MOON_DATA'
    df = _unload_pickle(Source)
    return df

def _Transform_Neg_Long(x):
    if x>=0:
        return x
    else:
        return 360+x
    
def Construct_DataFrame(Racine):            
    #On recupere chaque dataset separement
    F =  pd.read_csv(Racine + 'FFC_Final.txt')
    C1 = pd.read_csv(Racine + 'LU78287GT.csv' , encoding = 'ascii')
    C2 = pd.read_csv(Racine + 'Lunar_Impact_Crater_Database_v9Feb2009.csv')
    # On reformatte le nom des colonnes
    C2.columns = [''.join(f.split(' ')[1:]) for f in C2.columns]
    # On reformatte les nom pour qu il apparaisse
    C1['Name'] = [''.join(f.split('r:')[0].split(' ')) for f in map(str,C1.name.tolist())]
    C2['Name'] = [''.join(f.split(' ')) for f in C2.Name]
    # On met les nom en index
    C2.index = C2.Name
    F.index = F.Name
    # ON rassemble
    df = C1[['x','y','D[km]','Name']]
    df.columns = ['Long','Lat','Diameter','Name']
    # join FFC dataset
    F['Type'] = 1
    col_join = ['Class','Type']
    df = df.join(F[col_join] , on ='Name')
    # On ajoute les FFC qui etaient pas ds le dataset initiales
    df = df.append(F[~F.Name.isin(df.Name)][df.columns])
    # join C2 dataset
    col_join =  ['Age']
    df = df.join(C2[col_join] , on = 'Name')
    df.Class = df.Class.map(_Class_Type)
    df.Type = df.Type.map(_Class_Type)
    Ages = [f for f in map(str,list(set(df.Age.tolist()))) if f != 'nan']
    dict_ages = dict(zip(Ages,range(len(Ages))))
    dict_ages['nan'] = '19'
    df.Age = df.Age.map(lambda x:dict_ages[str(x)])
    
    df.index = np.arange( len( df ))
    df.Long = df.Long.map(_Transform_Neg_Long)
    return df
    
def Dataframe(Source):
    df =  _unload_pickle(Source)
    df.index = np.arange( len( df ) )
    df = df[ ( df.Diameter > 15 ) & ( df.Diameter < 180 ) ]
    return df

def df_feature(df,feat_df):
    for key in df.columns:
        feat_df[key] = row[key]
    return feat_df

def Initialize_feat_lola():
    feat_lola = dict.fromkeys(['L_ZInt','L_ZExt','L_ZRim'])
    for key,val in feat_lola.iteritems():
        feat_lola[key] = []
    return feat_lola

def Initialize_feat_df(df):
    feat_df = dict.fromkeys(df.columns)
    return feat_df
    
def Initialize_feat_grail(MapGrails):
    col = ['ZInt','ZExt']
    feat_grail  = []
    for MapGrail in MapGrails:
        feat_grail += [MapGrail.composante+'_'+f for f in col]

    feat_grail = dict.fromkeys(feat_grail)
    for key,val in feat_grail.iteritems():
        feat_grail[key] = []    
    return feat_grail

def df_update(feat,feat_tmp):
    for key in feat.iterkeys():
        if feat[key] is not None:
            feat[key] = np.vstack((feat[key],feat_tmp[key]))
        else:
            feat[key] = feat_tmp[key]
        
    return feat

def feat_update(feat,feat_tmp):
    for key in feat.iterkeys():
        feat[key].append(feat_tmp[key])
        
    return feat
            
###################
# PROGRAMME

###################
# Object MapLola et MapGrail
carte_lolas = Carte_Lola(Path_lola,pix)
MapGrails = Carte_Grail(Path_grail)
# MapGrails = [BinaryGrailTable(Path_grail+'34_12_3220_900_80_misfit_rad'),
#              BinaryGrailTable(Path_grail+'34_12_3220_900_80_misfit_theta')]

# on recupere le dataframe avec tous les craters
# Source = Root+'Data/CRATER_MOON_DATA'
Source = Root +'Data/'
df = Construct_DataFrame(Source)
# df = Crater_Data(Source)
# df = df[df.Name.isin(['Taruntius','Vitello','Hermite','Meton','A68'])]
# df = df[df.Name.isin(['Taruntius','Vitello'])]
df = df[ ( df.Diameter > 15 ) & ( df.Diameter < 180 ) ]
df = df.reindex(np.random.permutation(df.index))
# df = df[:25]

# Compteur
compteur_init = len(df)
compteur =  compteur_init
tracker = open('tracker_'+pix+'.txt','wr+')
tracker.write('Resolution de %s pixel par degree\n'%(str(pix)))
tracker.close()
# Variable utiles

failed = []
ind_border = []

# Initialisation dict et list
feat_df_tmp = Initialize_feat_df(df)
feat_df = Initialize_feat_df(df)
lola_tmp = Initialize_feat_lola()
lola = Initialize_feat_lola()
grail_tmp  = Initialize_feat_grail(MapGrails)
grail  = Initialize_feat_grail(MapGrails)

#Debut boucle
for carte_lola in carte_lolas:
    MapLola = BinaryLolaTable(carte_lola)
    border = MapLola.Boundary()
    dfmap  = df[(df.Long>border[0]) & (df.Long<border[1]) &(df.Lat>border[2]) & (df.Lat<border[3])]
    for i,row in dfmap.iterrows():
        # print 'Il reste encore %d/%d iterations \n'%(compteur,compteur_init)
        tracker = open('tracker_'+pix+'.txt','a')
        tracker.write('Il reste encore %d/%d iterations \n'%(compteur,compteur_init))
        tracker.close()

        Window_Coord = MapLola.Cylindrical_Window(1.3*row.Diameter/2.0,row.Lat,row.Long)
        if (Window_Coord[0]<border[0]) or (Window_Coord[1]>border[1])\
           or (Window_Coord[2]<border[2]) or (Window_Coord[3]>border[3]) or (np.isnan(Window_Coord).sum() !=0 ):
            ind_border.append(i)
        else:
            try:
                feat_df_tmp = df_feature(dfmap,feat_df_tmp)
                lola_tmp = Extract_Array_Lola(MapLola,row,lola_tmp)                
                for MapGrail in MapGrails:
                    key = MapGrail.composante
                    min_gravi,max_gravi = MapGrail.Global_Map_Stat()
                    grail_tmp = Extract_Array_Grail(MapGrail,row,grail_tmp,key)

                    # Mise a jour des dict
                feat_df = df_update(feat_df,feat_df_tmp)
                lola = feat_update(lola,lola_tmp)
                grail = feat_update(grail,grail_tmp)
            except:
                failed.append(i)
                
        compteur-=1

        #Pickle object
pickle_object = {'failed_border' : ind_border,
                 'failed_Error' : failed,
                 'feat_df' : feat_df,
                 'lola' : lola,
                 'grail' : grail}
with open(Output+'LOLA'+pix+'_GRAIL_Dataset_Raw', 'wb') as fi:
    pickle.dump(pickle_object, fi, pickle.HIGHEST_PROTOCOL)

