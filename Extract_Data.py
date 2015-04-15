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
platform = 'clavius'
pix = '4'
    
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
    
def Statistic(X):
    return {'mean':np.mean(X),
            'median':np.median(X),
            'std':np.std(X),
            'var':np.var(X),
            'max':np.max(X),
            'min':np.min(X)}

    
def Extract_Array_Lola(MapLola,row):
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
    h_ZI = ['LI_'+str(f) for f in range(len(Z_Int))]
    h_ZE = ['LE_'+str(f) for f in range(len(Z_Ext))]
    h_ZR = ['LR_'+str(f) for f in range(len(Z_Rim))]
    
    return (h_ZI,h_ZE,h_ZR),(Z_Int,Z_Ext,Z_Rim)

def Binned_Array_Lola(Z_Int,Z_Ext,Z_Couronne):

    # Profondeur par rapport a la preimpact surace
    Floor = Z_Ext.mean()-Z_Int
    depth_floor,bin_floor = np.histogram(Floor , range = (0, 3000), bins = 300)
    depth_floor = depth_floor / float(depth_floor.sum())    
    statistic_floor = Statistic(Floor)
    stat_floor = statistic_floor.values()
                
    # hauteur des rim par rapport a la preimpact surface
    Rim = Z_Couronne-Z_Ext.mean()
    height_rim,bin_rim = np.histogram(Rim, range = (-500, 2000), bins = 300)
    height_rim /= float(height_rim.sum())
    statistic_rim = Statistic(Rim)
    stat_rim = statistic_rim.values()

    # On remplie le header 
    h_stat_floor = ['F'+f for f in statistic_floor.keys()]
    h_depth_floor = ['DF_'+str(f) for f in range(len(depth_floor))]
    h_stat_rim = ['R'+f for f in statistic_rim.keys()]
    h_height_rim = ['R_'+str(f) for f in range(len(height_rim))]

    header = (h_stat_floor,h_depth_floor,h_stat_rim,h_height_rim)
    value = (stat_floor,depth_floor,stat_rim,height_rim)
    bins = ({'F':bin_floor,'R':bin_rim})
    
    return header , value , bins

def Binned_Array_Grail(Z_Int,Z_Ext,min_gravi,max_gravi):
    center = Z_Int-Z_Ext.mean()
    anomaly_center,bin_center = np.histogram(center,range = (-100,100),bins = 200)
    anomaly_center = anomaly_center / float(anomaly_center.sum())
    statistic_center = Statistic(center)
    stat_center = statistic_center.values()
    
    field = (MapGrail.name.split('/')[-1]).split('_')[6]
    h_center  = [field+'_GC_'+str(f) for f in range(len(anomaly_center))]
    h_stat_center = [field+'_'+f for f in statistic_center.keys()]

    header = (h_stat_center,h_center)
    value = (stat_center,anomaly_center)
    bins = ({field : bin_center})
    
    return header , value , bins
        
def Extract_Array_Grail(MapGrail,row):

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
    h_ZI = ['GI_'+str(f) for f in range(len(Z_Int))]
    h_ZE = ['GE_'+str(f) for f in range(len(Z_Int))]
    return (h_ZI,h_ZE) , (Z_Int,Z_Ext)

def _Class_Type(x):
    if np.isnan(x):
        return 0
    else:
        return x
        
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

    df.index = np.arange( len( df ) )
    df = df[ ( df.Diameter > 15 ) & ( df.Diameter < 180 ) ]
    return df
    
def Dataframe(Source):
    df =  _unload_pickle(Source)
    df.index = np.arange( len( df ) )
    df = df[ ( df.Diameter > 15 ) & ( df.Diameter < 180 ) ]
    return df

def df_feature(df):
    feature = np.array([row[f] for f in df.columns if f != 'Name'])
    h_feature = [f for f in df.columns if f != 'Name']
    return h_feature,feature

def update_grail(Feat_field,Feat):

    Feat[0] += (Feat_field[0],)
    Feat[1] += (Feat_field[1],)
    Feat[2] += (Feat_field[2],)
        
    return Feat[0],Feat[1],Feat[2]
    
###################
# PROGRAMME

###################
# Object MapLola et MapGrail
carte_lolas = Carte_Lola(Path_lola,pix)
MapGrails = Carte_Grail(Path_grail)

# on recupere le dataframe avec tous les craters
# Source = Root+'Data/CRATER_MOON_DATA'
Source = Root +'Data/'
df = Construct_DataFrame(Source)
# df = df[df.Name.isin(['Taruntius','Vitello'])]

# Compteur
compteur_init = len(df)
compteur =  compteur_init
tracker = open('tracker_'+pix+'.txt','wr+')
tracker.write('Resolution de %s pixel par degree\n'%(str(pix)))
tracker.close()
# Variable utiles
data = None
header = []
ind_border = []

#Debut boucle
for carte_lola in carte_lolas:
    MapLola = BinaryLolaTable(carte_lola)
    border = MapLola.Boundary()
    dfmap  = df[(df.Long>border[0]) & (df.Long<border[1]) &(df.Lat>border[2]) & (df.Lat<border[3])]
    for i,row in df.iterrows():
        print 'Il reste encore %d/%d iterations \n'%(compteur,compteur_init)
        tracker = open('tracker_'+pix+'.txt','a')
        tracker.write('Il reste encore %d/%d iterations \n'%(compteur,compteur_init))
        tracker.close()

        Window_Coord = MapLola.Cylindrical_Window(1.3*row.Diameter/2.0,row.Lat,row.Long)
        if (Window_Coord[0]<border[0]) or (Window_Coord[1]>border[1])\
           or (Window_Coord[2]<border[2]) or (Window_Coord[3]>border[3]) or (np.isnan(Window_Coord).sum() !=0 ):
            ind_border.append(i)
        else:
            h_df_feat , df_feat = df_feature(dfmap)
            h_Lola , Z = Extract_Array_Lola(MapLola,row)
            h_feat_lola , feat_lola , bin_lola = Binned_Array_Lola(Z[0],Z[1],Z[2])

            h_feat_grail, feat_grail,bin_grail = (),(),()
            for MapGrail in MapGrails:
                min_gravi,max_gravi = MapGrail.Global_Map_Stat()
                h_grail_field , Z_field = Extract_Array_Grail(MapGrail,row)
                h_feat_field, feat_field, bin_field = Binned_Array_Grail(Z_field[0],Z_field[1],min_gravi,max_gravi)
                h_feat_grail , feat_grail , bin_grail = update_grail([h_feat_field, feat_field, bin_field],[h_feat_grail, feat_grail, bin_grail])
                
                
            # On range tout dans un gros tableau
            if data == None:
                data = np.hstack(np.hstack((df_feat,feat_lola,np.hstack(feat_grail))))
            else:
                data = np.vstack((data,np.hstack(np.hstack((df_feat,feat_lola,np.hstack(feat_grail))))))
        compteur-=1

#Pickle object
header = np.hstack(np.hstack((h_df_feat,h_feat_lola,np.hstack(h_feat_grail))))
pickle_object = {'header': header,
                 'data' : data,
                 'bin_lola': bin_lola,
                 'bin_grail': bin_grail}
with open(Output+'LOLA'+pix+'_GRAIL_Dataset', 'wb') as fi:
    pickle.dump(pickle_object, fi, pickle.HIGHEST_PROTOCOL)



