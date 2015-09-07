# Objectif
# On creer une classe qui permet manipuler des binary tables

import numpy as np
import pickle
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import sys
from planetaryimage import PDS3Image
from matplotlib import cm
        
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
        ''' Return une carte avec 0<lon<360 et -90<lat<90'''
        X = self._Load_X()
        Y = self._Load_Y()
        Z = self._Load_Z()
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
        return (int(self.WESTERNMOST_LONGITUDE),
                int(self.EASTERNMOST_LONGITUDE),
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

    def Cylindrical_Window(self,radius,lat0,long0):

        # Passage en radian
        radi = radius*2*np.pi/(2*1734.4*np.pi)
        lamb0 = long0*np.pi/180.0
        phi0 = lat0*np.pi/180.0

        #Long/lat min (voir wikipedia)
        longll = -radi/np.cos(phi0)+lamb0
        latll = np.arcsin((-radi+np.sin(phi0)/np.cos(phi0))*np.cos(phi0))
        if np.isnan(latll):
          latll = -90*np.pi/180.0
        #Long/lat max (voir wikipedia)
        longtr = radi/np.cos(phi0)+lamb0
        lattr = np.arcsin((radi+np.tan(phi0))*np.cos(phi0))

        # print latll*180/np.pi,lat0,lattr*180/np.pi
        # print longll*180/np.pi,long0,longtr*180/np.pi
        return longll*180/np.pi,longtr*180/np.pi,latll*180/np.pi,lattr*180/np.pi
        
    def Extract_Grid(self,Radius,lat0,lon0):
        ''' Extrait une grille centre autour du crater
        dont les parametre sont fixe par Cadre '''
        
        Xa,Ya = np.meshgrid(self.X,self.Y)
        lon_m,lon_M,lat_m,lat_M = self.Cylindrical_Window(Radius,lat0,lon0)
        
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

    def Global_Map_Stat(self):
        return self.Z.min(),self.Z.max()
        
    def Circular_Mask(self,radius,radius_window,lat0,long0):
        ''' Radius correspond au rayon du cercle inscrit dans la fenetre
        radius window correspond au rayon du cercle inscrit dans la fenetre !'''
        
        # Extrait la fenetre qui correspond a cote = diametre
        X,Y,Z = self.Extract_Grid(radius_window,lat0,long0)
        lamb0 = long0*np.pi/180.0
        lamb = X*np.pi/180.0
        phi0 = lat0*np.pi/180.0
        phi = Y*np.pi/180.0
        radi = radius*2*np.pi/(2*1734.4*np.pi)        
        mask = (lamb-lamb0)**2*np.cos(phi0)**2+(np.sin(phi)/np.cos(phi0)-np.sin(phi0)/np.cos(phi0))**2<= (radi)**2
        return mask,Z[mask],Z[~mask]
    
    def Couronne_Mask(self,radius_int,radius_ext,lat0,long0):
        """ radius_int correspond au rayon interieur de la couronne
        radius_ext corresponds au rayon exterieur de la couronne """
        X,Y,Z = self.Extract_Grid(radius_ext,lat0,long0)
        mask_int,tmp,tmp = self.Circular_Mask(radius_int,radius_ext,lat0,long0)
        mask_ext,tmp,tmp = self.Circular_Mask(radius_ext,radius_ext,lat0,long0)
        # print mask_int & mask_ext
        # plt.pcolormesh(Z*(~mask_int&mask_ext))
        return Z[~mask_int & mask_ext]

    

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
        self.composante = self.img.split('_')[-1].split('.')[0]
        
    def Load_XYZ(self):
        ''' Return une carte avec -180<lon<180 et -90<lat<90'''
        X = self._Load_X()
        Y = self._Load_Y()
        Z = self._Load_Z()
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


class BinaryWACTable(PDS3Image,BinaryLolaTable):

    def __init__(self,which_wac):
        wac = PDS3Image.open(which_wac)
        for key,val in wac.__dict__.iteritems():
            setattr(self,key,val)

        for key,val in self.label['IMAGE_MAP_PROJECTION']:
            try :
                setattr(self,key,val.value)
            except:
                pass
        for key,val in self.label['IMAGE']:
            try :
                setattr(self,key,val)
            except:
                pass
        self.projection = str(self.label['IMAGE_MAP_PROJECTION']['MAP_PROJECTION_TYPE'])
        self.X,self.Y,self.Z = self.Load_XYZ()


    def _Load_X(self):
        ''' Info trouver dans cette base de donne
        http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/catalog/dsmap.cat'''

        Lon = np.arange(int(self.LINE_SAMPLES))+1

        if self.projection == 'EQUIRECTANGULAR':
            Lon = self.CENTER_LONGITUDE + (Lon - self.SAMPLE_PROJECTION_OFFSET -1)*self.MAP_SCALE*1e-3/(self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0))
        else:
            print 'Projection pas implementer, implementer d abord'
            sys.exit()                  

        return Lon*180/np.pi
        
    def _Load_Y(self):
        ''' Info trouver dans cette base de donne
        http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/CATALOG/DSMAP.CAT
        Erreur dans EQUIRECTANGULAR PROJECTIOn pour LAT +L0 au lieu de -'''

        Lat = np.arange(int(self.LINES))+1

        if self.projection == 'EQUIRECTANGULAR':
            Lat = ((1 + self.LINE_PROJECTION_OFFSET- Lat)*self.MAP_SCALE*1e-3/self.A_AXIS_RADIUS)
        else:
            print 'Projection pas implementer, implementer d abord'
            sys.exit()                  

        return Lat*180/np.pi

    def Load_XYZ(self):
        ''' Return une carte avec 0<lon<360 et -90<lat<90'''
        X = self._Load_X()
        Y = self._Load_Y()
        Z = self.data
        return X,Y,Z
        
    def Plot_Global_Map(self):
        m = Basemap(projection='moll',lon_0=0,resolution='c')
        fig = plt.figure(figsize=(15,7.5))
        ax = fig.add_subplot(111)
        X,Y = np.meshgrid(self.X,self.Y)
        m.pcolormesh(X,Y,self.Z,latlon = True, cmap ='gray')

        
class Crater(object):
            
    def __init__(self,Name):
        self.Name = Name
        df = self.Crater_Data()
        df = df[df.Name == Name]
        print df
        [setattr(self,f,float(df[f])) for f in df.columns if f not in ['Name']]
        self.Taille_Window = 0.8*self.Diameter
        if self.Long <0.0:
            self.Long = 360+self.Long
        
    def Crater_Data(self):
        Racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/Data/'
        Source = 'CRATER_MOON_DATA'
        df = self._unload_pickle(Racine+Source)
        print df.columns
        return df
        
    def _unload_pickle(self,input_file):
        with open(input_file, 'rb') as f:
            df = pd.read_pickle(input_file)
        return df

    def Plot_Mesh(self):
        fig = plt.figure(figsize=(12,7.5))
        ax = fig.add_subplot(111)
        X,Y,Z = self.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        ax.pcolormesh(X,Y,Z, cmap ='binary')
        ax.scatter(self.Long,self.Lat)
        return fig,ax

    def Plot_Mesh2(self,Wac,Grail):
        fig = plt.figure(figsize=(12,7.5))
        ax = fig.add_subplot(111)
        Xw,Yw,Zw = Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        Xg,Yg,Zg = Grail.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        ax.pcolormesh(Xw,Yw,Zw, cmap ='binary')
        ax.pcolormesh(Xg,Yg,Zg, cmap ='jet')
        return fig,ax

    def plot_LOLA(self,Wac,Lola):
        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(111)
        X,Y,Z = Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolor(X,Y,Z,cmap = cm.gray ,ax  = ax1)
        Xl,Yl,Zl = Lola.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        Xl,Yl = m(Xl,Yl)
        CS3 = m.contourf(Xl,Yl,Zl,cmap='gist_earth',animated=True , alpha = 0.1)
        cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        cb.set_label('Topography', size = 24)
        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=100,marker ='v')

    def plot_GRAIL(self,Wac,Grail):
        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(111)
        X,Y,Z = Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolor(X,Y,Z,cmap = cm.gray ,ax  = ax1)
        Xg,Yg,Zg = Grail.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        clevs = np.arange(-50,50,5)
        Xg,Yg = m(Xg,Yg)
        CS2 = m.contour(Xg,Yg,Zg,clevs,linewidths=0.5,colors='k',animated=True)
        CS3 = m.contourf(Xg,Yg,Zg,clevs,cmap=plt.cm.RdBu_r,animated=True , alpha = 0.5)
        cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        cb.set_label('Gravity anomaly',size = 24)
        
    def plot_tout(self,Wac,Grail,Lola):
        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(121)
        X,Y,Z = Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolor(X,Y,Z,cmap = cm.gray ,ax  = ax1)
        Xl,Yl,Zl = Lola.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        Xl,Yl = m(Xl,Yl)
        CS2 = m.contour(Xl,Yl,Zl,linewidths=0.5,colors='k',animated=True)
        CS3 = m.contourf(Xl,Yl,Zl,cmap='gist_earth',animated=True , alpha = 0.1)
        cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        cb.set_label('Topography', size = 24)

        
        ax2 = fig.add_subplot(122)
        m.pcolor(X,Y,Z,cmap = cm.gray ,ax  = ax2)
        Xg,Yg,Zg = Grail.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        clevs = np.arange(-100,100,10)
        Xg,Yg = m(Xg,Yg)
        CS2 = m.contour(Xg,Yg,Zg,clevs,linewidths=0.5,colors='k',animated=True)
        CS3 = m.contourf(Xg,Yg,Zg,clevs,cmap=plt.cm.RdBu_r,animated=True , alpha = 0.3)
        cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        cb.set_label('Gravity anomaly',size = 24)
        
    def Deg(self,radius):
        return radius*360/(2*np.pi*1734.4)

        
        




        


