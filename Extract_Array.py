# Objectif
# On creer une classe qui permet manipuler des binary tables

import numpy as np
import pickle
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import window

class BinaryLolaTable(object):

    def __init__(self,which_lola):
        ''' Parameter
        self.img : nom du fichier
        self.lbl : '''
        self.name = which_lola
        self.img = which_lola + '.img'
        self.lbl = which_lola + '.lbl'
        self.X, self.Y, self.Z = self.Load_XYZ()
        
    def _Load_Info_LBL(self):
        with open(self.lbl, 'r') as f:
            for line in f:
                attr = [f.strip() for f in line.split('=')]
                if len(attr) == 2:
                    setattr(self,attr[0],attr[1].split(' ')[0])
                    
    def Load_XYZ(self):
        ''' Return une carte avec -180<lon<180 et -90<lat<90'''
        X = self._Load_X()
        Y = self._Load_Y()
        Z = self._Load_Z()
        Z = np.hstack((Z[:,int(Z.shape[1]/2):],Z[:,:int(Z.shape[1]/2)]))
        X = X-180.0
        return X,Y,Z
        
    def _Load_X(self):
        ''' Info trouver dans cette base de donne
        http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/catalog/dsmap.cat'''
        if self.img == '' or self.lbl == '' :
            raise Exception
        self._Load_Info_LBL()
        Lon = np.arange(int(self.LINE_SAMPLES))+1
        Lon = float(self.CENTER_LONGITUDE) + (Lon-float(self.SAMPLE_PROJECTION_OFFSET) -1)/ float(self.MAP_RESOLUTION)
        return Lon
        
    def _Load_Y(self):
        ''' Info trouver dans cette base de donne
        http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/catalog/dsmap.cat'''
        if self.img == '' or self.lbl == '' :
            raise Exception        
        self._Load_Info_LBL()
        Lat = np.arange(int(self.LINES))+1
        Lat =  float(self.CENTER_LATITUDE) - (Lat-float(self.LINE_PROJECTION_OFFSET) -1)/ float(self.MAP_RESOLUTION)
        return Lat
        
    def _Load_Z(self):
        if self.img == '' or self.lbl == '' :
            raise Exception
        self._Load_Info_LBL()
        Z = np.fromfile(self.img, dtype = 'int16')*float(self.SCALING_FACTOR)
        Z = np.reshape(Z,(int(self.LINES),int(self.LINE_SAMPLES)))
        return Z

    def Boundary(self):
        self._Load_Info_LBL()
        print 
        return (int(self.WESTERNMOST_LONGITUDE)-180,
                int(self.EASTERNMOST_LONGITUDE)-180,
                int(self.MINIMUM_LATITUDE),
                int(self.MAXIMUM_LATITUDE))

    def kp_func(self,lat,lon,lat0,long0):
        kp = float(1.0) + np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-long0)
        kp = np.sqrt(float(2)/kp)
        return kp
        
    def Lambert_Window(self,radius,lat0,long0):

        radius = radius*360.0/(np.pi*2*1734.4)
        radius = radius*np.pi / 180.0
        lat0 = lat0*np.pi/180.0
        long0 = long0*np.pi/180.0

        bot = self.kp_func(lat0-radius,long0,lat0,long0)
        bot = bot * ( np.cos(lat0)*np.sin(lat0-radius) - np.sin(lat0)*np.cos(lat0-radius) )
        x = bot
        y = bot
        rho = np.sqrt(x**2 + y**2)
        c = 2.0 * np.arcsin(rho/float(2.0))
        latll = np.arcsin(np.cos(c)*np.sin(lat0)  + y*np.sin(c)*np.cos(lat0)/rho ) * float(180.0) / np.pi
        lon = long0  + np.arctan2(x*np.sin(c), rho*np.cos(lat0)*np.cos(c) - y*np.sin(lat0)*np.sin(c))
        longll = lon * 180.0 / np.pi
	
        x = -bot
        y = -bot
        rho = np.sqrt(x**2 + y**2)
        c = 2.0 * np.arcsin(rho/2.0)
        lattr =np.arcsin(np.cos(c)*np.sin(lat0)  + y*np.sin(c)*np.cos(lat0)/rho ) * float(180.0) / np.pi
        lon = long0  + np.arctan2(x*np.sin(c), rho*np.cos(lat0)*np.cos(c) - y*np.sin(lat0)*np.sin(c))
        longtr = lon * 180.0 / np.pi

        return longll,longtr,latll,lattr
            
    def Extract_Grid(self,Taille,lat0,lon0):
        ''' Extrait une grille centre autour du crater
        dont les parametre sont fixe par Cadre '''
        
        Xa,Ya = np.meshgrid(self.X,self.Y)
        lon_m,lon_M,lat_m,lat_M = self.Lambert_Window(Taille,lat0,lon0)

        X_Ok = self.X[np.where((self.X>= lon_m) & (self.X <= lon_M))]
        Y_Ok = self.Y[np.where((self.Y >= lat_m) & (self.Y <= lat_M))]
        Size_X = X_Ok.shape[0]
        Size_Y = Y_Ok.shape[0]        

        Za = self.Z[np.where((Xa >= lon_m) & (Xa <= lon_M) & (Ya >= lat_m) & (Ya <= lat_M))]
        Za = np.reshape(Za,(Size_Y,Size_X))
        Xa,Ya = np.meshgrid(X_Ok,Y_Ok)

        return Xa,Ya,Za

    def Plot_Global_Map(self):
        m = Basemap(llcrnrlon =-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = 0,lon_0 = 0)
        X,Y = np.meshgrid(self.X,self.Y)
        X,Y = m(X,Y)
        plt.pcolormesh(X,Y,self.Z)


    def Circular_Mask(self,Int_D,Ext_D,lat0,long0):
        ''' Int_D correspond au rayon du crater que l'on considere
        Ext_D correspond a bord de la fenetre que l'on veut'''
        
        # Extrait la fenetre qui correspond a cote = diametre
        X,Y,Z = self.Extract_Grid(Ext_D,lat0,long0)
        lon_m,lon_M,lat_m,lat_M = self.Lambert_Window(Int_D,lat0,long0)
        d_x=lon_M-long0
        d_y=lat_M-lat0
        mask = ((X-long0)/d_x)**2+((Y-lat0)/d_y)**2 <=1
        return Z[mask],Z[~mask]
    
    def Couronne_Mask(self,int_D,ext_D,lat_0,long_0):
        """ Int_D correspond au rayon interieur de la couronne et ext_D au rayon
        exterieur de la couronne """
        X,Y,Z = self.Extract_Grid(ext_D,lat_0,long_0)
        long_m_Int,long_M_Int,lat_m_Int,lat_M_Int = self.Lambert_Window(int_D,lat_0,long_0)
        long_m_Ext,long_M_Ext,lat_m_Ext,lat_M_Ext = self.Lambert_Window(ext_D,lat_0,long_0)
        d_x_Int = long_M_Int-long_0
        d_y_Int = lat_M_Int-lat_0
        d_x_Ext = long_M_Ext-long_0
        d_y_Ext = lat_M_Ext-lat_0
        mask = (((X-long_0)/d_x_Int)**2+((Y-lat_0)/d_y_Int)**2 >=1) & (((X-long_0)/d_x_Ext)**2+((Y-lat_0)/d_y_Ext)**2 <=1)

        return Z[mask],Z[~mask]

class BinaryGrailTable(BinaryLolaTable):
    """Images are BINARY raster images (0 to 360 E, and 90N to 90S), with 
    - Nlongitude = 5761
    - Nlatitude = 2881
    - interval = 0.0625 degrees
    each binary entry is a 4 byte (single precission) floating point numbe"""
    
    def __init__(self,which_grad):
        self.name = which_grad
        self.img = which_grad+'.dat'
        self.X,self.Y,self.Z = self.Load_XYZ()
        
    def Load_XYZ(self):
        ''' Return une carte avec -180<lon<180 et -90<lat<90'''
        X = self._Load_X()
        Y = self._Load_Y()
        Z = self._Load_Z()
        Z = np.hstack((Z[:,int(Z.shape[1]/2):],Z[:,:int(Z.shape[1]/2)]))
        X = X-180.0
        return X,Y,Z
        
    def _Load_X(self):
        if self.img == '':
            raise Exception
        Lon = np.arange(0,360.0625,0.0625)
        return Lon
        
    def _Load_Y(self):
        ''' Info trouver dans cette base de donne
        http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/catalog/dsmap.cat'''
        if self.img == '':
            raise Exception
        Lat = np.arange(90,-90.0625,-0.0625)
        return Lat
        
    def _Load_Z(self):
        if self.img == '':
            raise Exception
        Z = np.fromfile(self.img, dtype = 'float32')
        Z = np.reshape(Z,(2881,5761))
        return Z
        
    def Plot_Global_Map(self):
        m = Basemap(llcrnrlon =-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = 0,lon_0 = 0)
        X,Y = np.meshgrid(self.X,self.Y)
        # X,Y = m(X,Y)
        plt.pcolormesh(X,Y,self.Z)

    
class Crater(BinaryLolaTable):
            
    def __init__(self,Name):
        super(Crater,self).__init__()
        self.Name = Name
        df = self.Crater_Data()
        df = df[df.Name == Name]
        [setattr(self,f,float(df[f])) for f in df.columns if f not in ['Name']]
        self.Taille_Window = 0.8*self.Diameter

    def Crater_Data(self):
        Racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/Data/'
        Source = 'CRATER_MOON_DATA'
        df = self._unload_pickle(Racine+Source)
        return df
        
    def _unload_pickle(self,input_file):
        with open(input_file, 'rb') as f:
            df = pd.read_pickle(input_file)
        return df

    def Plot_Mesh(self):
        X,Y,Z = self.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        plt.pcolormesh(X,Y,Z)
        plt.scatter(self.Long,self.Lat)

    def Plot_Basemap(self):
        X,Y,Z = self.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = self.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        plt.pcolormesh(X,Y,Z)
        lon,lat = m(self.Long,self.Lat)
        plt.scatter(lat,lon)
        
        




        


