import numpy as np
import pandas as pd
import os,sys
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from Data_utils import *
import time

# Fonction

def LongRecap(elt):
    if elt<0:
        return 360+elt
    else:
        return elt


def Maria_Location_3(root):

    # Return two vector, one containing the lattitude for the maria, and another containiung
    # the longitude for the maria
    
    import numpy as np

    source = os.path.join(root,'USGS_mare_basalts_2.gmt')

    Lat = []
    Long = []
    Bool= False
    List=[]
    l=[]
    with open(source, 'r') as file:
        for line in file :
            if line.strip().split(' ')[1] == '-G0' :
                Bool = True
                List.append(l)
                l=[]
            elif line.strip().split(' ')[1] == '-G255' :
                Bool = False

            if Bool:
                if line.strip().split(' ')[0] == '>':
                    pass
                else:
                    Long = (np.float(line.strip().split(' ')[0]))
                    Lat = (np.float(line.strip().split(' ')[1]))
                    l.append((LongRecap(Long),Lat))
    return List


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


    
# Fichier de base !
root = '/Users/thorey/Documents/These/Projet/FFC/Classification/'
Output = os.path.join(root,'Data/')


Maria = MultiPolygon(map(Polygon,Maria_Location_3(Output)[1:])).buffer(0)

for pix in [4,16,64]:

    print 'pix : ', pix

    data = Data(pix,Output,'').data
    df = pd.DataFrame(np.hstack(data['feat_df'].values()),columns = data['feat_df'].keys()).convert_objects(convert_numeric= True)
    filtre = df.Type == 1.0
    FFC = df[filtre]

    print 'Mask_C_FFC'
    tic = time.time()
    df['Mask_C_FFC'] = [haversine(FFC.Long.map(float)*np.pi/180.0,FFC.Lat.map(float)*np.pi/180.0,float(df.Long[i])*np.pi/180.0,float(df.Lat[i])*np.pi/180.0).min()<150
                     for i in range(len(df.Lat))]
    toc = time.time()
    print 'Mask_C_FFC', toc-tic

    print 'Mask_SPA_Out'
    tic = time.time()
    df['Mask_SPA_Out'] = [haversine((-168.9)*np.pi/180.0,-55.0*np.pi/180.0,df.Long[i]*np.pi/180.0,df.Lat[i]*np.pi/180.0)>
                       1.238*970.0 for i in range(len(df.Lat))]
    toc = time.time()
    print 'Mask_SPA_Out', toc-tic

    print 'Mask_Highland'
    tic = time.time()
    df['Mask_Highlands'] = [not Maria.contains(Point(df.Long[i],df.Lat[i])) for i in range(len(df.Lat))]
    toc = time.time()
    print 'Mask_Highland', toc-tic

    print 'index'
    tic = time.time()
    df['Index'] = range(len(df))
    toc = time.time()
    print 'index', toc-tic    
    
    data['feat_df'] = {k: np.array(v.values())[:,np.newaxis] for k,v in df.to_dict().iteritems()}
    with open(Output+'LOLA'+str(pix)+'_GRAIL_Dataset_2', 'wb') as fi:
        pickle.dump(data, fi, pickle.HIGHEST_PROTOCOL)

