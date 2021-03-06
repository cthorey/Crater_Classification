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
import pdb
import os
from Extract_Array import *
import scipy.stats as st
from shapely.geometry import *

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

def Binned_Array_Lola(Z_Int,Z_Ext,Z_Couronne,h_feat_lola,feat_lola):

    # Profondeur par rapport a la preimpact surace
    Floor = Z_Ext.mean()-Z_Int
    depth_floor,bin_floor = np.histogram(Floor , np.arange(0,4000,10))
    statistic_floor = Statistic(Floor)
    stat_floor = statistic_floor.values()
                
    # hauteur des rim par rapport a la preimpact surface
    Rim = Z_Couronne-Z_Ext.mean()
    height_rim,bin_rim = np.histogram(Rim, np.arange(0,2000,10))
    statistic_rim = Statistic(Rim)
    stat_rim = statistic_rim.values()

    # On remplie le header 
    h_stat_floor = ['F'+f for f in statistic_floor.keys()]
    h_depth_floor = ['DF_'+str(f) for f in np.arange(len(depth_floor))]
    h_stat_rim = ['R'+f for f in statistic_rim.keys()]
    h_height_rim = ['R_'+str(f) for f in np.arange(len(height_rim))]

    # header = (h_stat_floor,h_depth_floor,h_stat_rim,h_height_rim)
    # value = (np.array(stat_floor,dtype=float),np.array(depth_floor),np.array(stat_rim),np.array(height_rim))

    # On met a jour les dictionnaires
    if feat_lola['stat_floor'] == None:
        h_feat_lola = {'h_stat_floor':h_stat_floor,
                       'h_depth_floor':bin_floor,
                       'h_stat_rim':h_stat_rim,
                       'h_height_rim': bin_rim}

    feat_lola['stat_floor'] = np.array(stat_floor,dtype=float)
    feat_lola['depth_floor'] = np.array(depth_floor,dtype=float)
    feat_lola['stat_rim'] = np.array(stat_rim,dtype=float)
    feat_lola['height_rim'] = np.array(height_rim,dtype=float)
    
    
    return h_feat_lola , feat_lola 

def Binned_Array_Grail(Z_Int,Z_Ext,min_gravi,max_gravi,feat_grail,h_feat_grail,key):
    center = Z_Int-Z_Ext.mean()
    anomaly_center,bin_center = np.histogram(center,np.arange(-100,100,2.5))
    statistic_center = Statistic(center)
    stat_center = np.array(statistic_center.values(),dtype=float)
    
    field = (MapGrail.name.split('/')[-1]).split('_')[6]
    h_center  = [field+'_GC_'+str(f) for f in range(len(anomaly_center))]
    h_stat_center = [field+'_'+f for f in statistic_center.keys()]

    h_feat_grail['h_'+key+'_stat_center'] = h_stat_center
    h_feat_grail['h_'+key+'_anomaly_center'] = bin_center
    feat_grail[key+'_stat_center'] = np.array(stat_center)
    feat_grail[key+'_anomaly_center'] = np.array(anomaly_center)
    
    
    return h_feat_grail , feat_grail 
        
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
        
def Crater_Data(Source):
    Source = Source+'CRATER_MOON_DATA'
    df = _unload_pickle(Source)
    return df

def _Transform_Neg_Long(x):
    if x>=0:
        return x
    else:
        return 360+x

def Age_Scale(Age):
    Dict_Name = {'coper' : 6, # Copernican plus recent
                 'erato' : 5, # Eratothetian 
                 'imbri' : 3, # Lower imbriam ( Il y en a qu'un ou deux qui n'a pas de lower ou upper -> Lower'')
                 'lower' : 3, # Lower imbriam
                 'upper' : 4, # Upper Imbriam
                 'necta' : 2, # Nectarian
                 'pre-n' : 1, # Pre-nectarian
                 'nan':7}
    return Dict_Name[Age]
    
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

    # Join C2 on Name
    col_join =  ['Age']
    df = df.join(C2[col_join] , on = 'Name')
    df.Class = df.Class.map(_Class_Type)
    df.Type = df.Type.map(_Class_Type)
    df.Age = df.Age.map(lambda x:Age_Scale(str(x).strip().lower()[:5]))
    df.index = np.arange( len( df ))
    df.Long = df.Long.map(_Transform_Neg_Long)
    df.index = range(len(df))
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

def update_grail(Feat_field,Feat):

    Feat[0] += (Feat_field[0],)
    Feat[1] += (Feat_field[1],)
    Feat[2] += (Feat_field[2],)
        
    return Feat[0],Feat[1],Feat[2]

def Initialize_feat_lola():
    h_feat_lola = dict.fromkeys(['stat_floor','depth_floor','stat_rim','height_rim'])
    feat_lola = dict.fromkeys(['stat_floor','depth_floor','stat_rim','height_rim'])
    return h_feat_lola , feat_lola

def Initialize_feat_df(df):
    feat_df = dict.fromkeys(df.columns)
    return feat_df

def haversine(lon1, lat1, lon2, lat2):

    import numpy as np
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees). The angle
    should be in rad
    """
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 1734 * c
    return km
        
def Initialize_feat_grail(MapGrails):
    col = ['stat_center','anomaly_center']
    feat_grail  = []
    h_feat_grail = []
    for MapGrail in MapGrails:
        feat_grail += [MapGrail.composante+'_'+f for f in col]
        h_feat_grail += ['h_'+MapGrail.composante+'_'+f for f in col]

    feat_grail = dict.fromkeys(feat_grail)
    h_feat_grail = dict.fromkeys(h_feat_grail)

    return h_feat_grail, feat_grail

def feat_update(feat,feat_tmp):
    for key in feat.iterkeys():

        if feat[key] is not None:
            feat[key] = np.vstack((feat[key],feat_tmp[key]))
        else:
            feat[key] = feat_tmp[key]
        
    return feat
            
###################
# PROGRAMME

###################
# Object MapLola et MapGrail
    
carte_lolas = Carte_Lola(Path_lola,pix)
# MapGrails = Carte_Grail(Path_grail)
MapGrails = [BinaryGrailTable(Path_grail+'34_12_3220_900_80_misfit_rad'),
             BinaryGrailTable(Path_grail+'34_12_3220_900_80_misfit_theta')]
print len(carte_lolas),Path_lola

sys.exit()
# on recupere le dataframe avec tous les craters
Source = Root +'Data/'
df = Construct_DataFrame(Source)

# df = Crater_Data(Source)
# df = df[df.Name.isin(['Taruntius','Vitello','Hermite','Meton','A68'])]
# df = df[df.Name.isin(['Taruntius','Vitello'])]
df = df[ ( df.Diameter > 15 ) & ( df.Diameter < 180 ) ]
df = df.reindex(np.random.permutation(df.index))

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
h_feat_lola_tmp , feat_lola_tmp = Initialize_feat_lola()
h_feat_lola , feat_lola = Initialize_feat_lola()
h_feat_grail_tmp , feat_grail_tmp  = Initialize_feat_grail(MapGrails)
h_feat_grail , feat_grail  = Initialize_feat_grail(MapGrails)

#Debut boucle
for carte_lola in carte_lolas:
    MapLola = BinaryLolaTable(carte_lola)
    border = MapLola.Boundary()
    dfmap  = df[(df.Long>border[0]) & (df.Long<border[1]) &(df.Lat>border[2]) & (df.Lat<border[3])]
    for i,row in dfmap.iterrows():
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
                h_Lola , Z = Extract_Array_Lola(MapLola,row)
                h_feat_lola , feat_lola_tmp  = Binned_Array_Lola(Z[0],Z[1],Z[2],h_feat_lola,feat_lola_tmp)
                
                for MapGrail in MapGrails:
                    key = MapGrail.composante
                    min_gravi,max_gravi = MapGrail.Global_Map_Stat()
                    h_grail_field , Z_field = Extract_Array_Grail(MapGrail,row)
                    h_feat_grail, feat_grail_tmp = Binned_Array_Grail(Z_field[0],Z_field[1],min_gravi,max_gravi,feat_grail_tmp,h_feat_grail,key)

                    # Mise a jour des dict
                feat_df = feat_update(feat_df,feat_df_tmp)
                feat_lola = feat_update(feat_lola,feat_lola_tmp)
                feat_grail = feat_update(feat_grail,feat_grail_tmp)
            except:
                failed.append(i)
            
    compteur-=1
    pickle_object = {'failed_border' : ind_border,
                     'failed_Error' : failed,
                     'feat_df' : feat_df,
                     'feat_lola' : feat_lola,
                     'h_feat_lola' : h_feat_lola,
                     'feat_grail' : feat_grail,
                     'h_feat_grail' : h_feat_grail}
    with open(Output+'LOLA'+pix+'_GRAIL_Dataset', 'wb') as fi:
        pickle.dump(pickle_object, fi, pickle.HIGHEST_PROTOCOL)


#Pickle object
pickle_object = {'failed_border' : ind_border,
                 'failed_Error' : failed,
                 'feat_df' : feat_df,
                 'feat_lola' : feat_lola,
                 'h_feat_lola' : h_feat_lola,
                 'feat_grail' : feat_grail,
                 'h_feat_grail' : h_feat_grail}
with open(Output+'LOLA'+pix+'_GRAIL_Dataset', 'wb') as fi:
    pickle.dump(pickle_object, fi, pickle.HIGHEST_PROTOCOL)



