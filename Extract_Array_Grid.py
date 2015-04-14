###############
# Fonction necessaire a l'extraction'

def Lambert_Equal_Area(lat,lon,lat0,lon0,size):

    import numpy as np
    import window
    # Cadrage
    Fen=str(window.lambert((float(size)*360/(10917.0)),lat0,lon0)[4])
    Fen=[elt for elt in Fen.split() if elt!=''][0]
    Fen=Fen[1:-1].split('/')
    long_Min=float(Fen[0])
    long_Max=float(Fen[2])

    # if long_Min<0 :
    #     long_Min=360+long_Min
    # if long_Max<0:
    #     long_Max=360+long_Max
    lat_Min=float(Fen[1])
    lat_Max=float(Fen[3]) 

    # CONVERSION X,Y
    phi0 = lat0*np.pi/180.0
    lamb0 = lon0*np.pi/180.0
    phi = lat*np.pi/180.0
    lamb = lon*np.pi/180.0
    
    alpha = 1.0+np.cos(lamb-lamb0)+np.cos(lamb0)*np.cos(lamb)*(np.cos(phi-phi0)-1.0)
    alpha = np.sqrt(2/alpha)
    X = alpha*np.cos(lamb)*np.sin(phi-phi0)
    Y = alpha*(np.sin(lamb-lamb0)-np.sin(lamb0)*np.sin(lamb)*(np.cos(phi-phi0)-1.0))

    # Conversion x,y
    top = long_Max*np.pi/180.0
    bottom = long_Min*np.pi/180.0
    left = lat_Min*np.pi/180.0
    right = lat_Max*np.pi/180.0

    x = 100.0*(0.5 +(180/(np.pi*np.cos(phi0)*(right-left)))*X)
    y = 100.0*(0.5 -(180/(np.pi*(top-bottom)))*Y)
    return X,Y
    
def Extract_data(df,Path,Carte):

    import numpy as np
    import pandas as pd
    import matplotlib.pylab as plt
    import sys
    
    ################################################
    # Load de la carte sous forme de 3 arrays
    # x_d correspond a un vecteur qui range les longitudes
    # y_d correspond a un vecteur qui range les latitudes
    # z_d correspond a une grosse matrice qui prend les valeurs du champ de gravite. Il est index par x_d en collone et y_d en ligne.

    Carte = Carte+'.grd'
    x_d,y_d,z_d = Sample_Grd(Path,Carte)
    diameter = float(df.Diameter) # Diametre du crater
    outer_diameter = float(df.Diameter) # Defini le rayon ext de la zone
    lat_0 = float(df.Lat)
    long_0 = float(df.Long)
    print diameter,lat_0,long_0
    x,y,z = Select_Zoom(x_d,y_d,z_d,diameter,lat_0,long_0)
    # try:
    #     # Select la restriction de l'enorme matrice a une fenetre carrer centre sur le
    #     # centre du crater (lat0,long0) dans lequelle est inscrit un cercle de diametre size_e.
    #     x,y,z = Select_Zoom(x_d,y_d,z_d,diameter,lat_0,long_0)
    # except:
    #     return None

    plt.subplot(1,2,1)
    x_a,y_a = np.meshgrid(x,y)
    plt.pcolormesh(x_a,y_a,np.array(z))
    plt.subplot(1,2,2)
    x_a,y_a = Lambert_Equal_Area(x_a,y_a,lat_0,long_0,outer_diameter)
    plt.pcolormesh(x_a,y_a,np.array(z))
    plt.show()
    return np.array(z)

def Select_Zoom(x,y,z,size,lat_0,long_0):
    
    import window
    import numpy as np
    import matplotlib.pylab as plt
    import sys
    
    Fen=str(window.lambert((float(size)*360/(10917.0)),lat_0,long_0)[4])
    Fen=[elt for elt in Fen.split() if elt!=''][0]
    Fen=Fen[1:-1].split('/')
    long_Min=float(Fen[0])
    long_Max=float(Fen[2])

    if long_Min<0 :
        long_Min=360+long_Min
    if long_Max<0:
        long_Max=360+long_Max


    lat_Min=float(Fen[1])
    lat_Max=float(Fen[3])
    if long_Min>long_Max:
        data_zoom_x1=x[(x.long>long_Min)]
        data_zoom_x2=x[(x.long<long_Max)]
        data_zoom_x=data_zoom_x1.append(data_zoom_x2)
    else:
        data_zoom_x=x[(x.long>long_Min) & (x.long<long_Max)]
    data_zoom_y=y[(y.lat>lat_Min) & (y.lat<lat_Max)]
    data_zoom=z.iloc[data_zoom_y.index.tolist(),data_zoom_x.index.tolist()]
    
    return data_zoom_x,data_zoom_y,data_zoom

def Sample_Grd(Path,Carte):
    
    import numpy as np
    from scipy.io import netcdf_file as netcdf
    from Scientific.IO.NetCDF import NetCDFFile
    import scipy as sp
    import pandas as pd
    import sys
    import os

    file=os.listdir(Path)
    grdfile=Path+'/'+Carte
    print grdfile

    if Carte == 'LDEM_64.grd':
        xf = 'x'
        yf = 'y'
    else:
        xf ='lon'
        yf = 'lat'
        
    x=pd.DataFrame(np.copy(NetCDFFile(grdfile,'r').variables[xf][::],'f4'),columns=['long'],dtype='f4')
    y=pd.DataFrame(np.copy(NetCDFFile(grdfile,'r').variables[yf][::],'f4'),columns=['lat'],dtype='f4')
    z=pd.DataFrame(np.copy(NetCDFFile(grdfile,'r').variables['z'][::],'f4'),columns=x.index,index=y.index,dtype='f4')
    

    return x,y,z 
