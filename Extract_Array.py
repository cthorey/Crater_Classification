# Objectif
# On creer une classe qui permet manipuler des binary tables

import numpy as np
import pickle
import pandas as pd
import pylab as plt
from mpl_toolkits.basemap import Basemap
import sys
from planetaryimage import PDS3Image
from matplotlib import cm
import os
from Data_utils import *

        
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
            
    def __init__(self,ide,idx,racine):
        '''n pour le designer par son nom et i pour le designer par son
        index '''

        self.racine = racine
        inde = {'n':'Name','i':'Index'}
        df = self.Crater_Data()
        df = df[df[inde[idx]] == ide]
        if len(df) == 0:
            print 'Correpond a aucun crater'
            raise Exception
        [setattr(self,f,float(df[f])) for f in df.columns if f not in ['Name','Index']]
        self.Taille_Window = 0.6*self.Diameter
        if self.Long <0.0:
          self.Long = 360+self.Long

        self.Wac = ''
        #self.Load_Wac('128')
        #self.Load_Lola('ldem_16')
        #self.Load_Grail('34_12_3220_900_80_misfit_rad')
        
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

        return longll*180/np.pi,longtr*180/np.pi,latll*180/np.pi,lattr*180/np.pi

    def _format_lon(self,lon):
        lonf = float((lon//90+1)*90-45)        
        st = str(lonf).split('.')
        loncenter =''.join(("{0:0>3}".format(st[0]),st[1]))

        return lonf,loncenter

    def _format_lat(self,lat):
        if lat<0:
            latcenter = '450S'
            latf = -45.0
        else:
            latcenter = '450N'
            latf = 45.0
        return latf,latcenter

    def Define_Case(self,lon_m,lon_M,lat_m,lat_M):
        ''' Case 1 : 0 Pas d'overlap, crater contenu ds une carte
        Case 2: 1Overlap au niveau des long
        Case 3:2 OVerlap au niveau des lat
        Case 4 :3 Overlap partout
        Boolean : True si overlap, false sinon'''
        
        map_lonmax = (lon_M//90+1)*90-45
        map_lonmin = (lon_m//90+1)*90-45
        lonBool = map_lonmax != map_lonmin
        latBool = lat_m*lat_M < 0

        if not lonBool and not latBool:
            print 'Cas1'
            self.Cas_1(lon_m,lat_m)
        elif lonBool and not latBool:
            print 'Cas2'
            self.Cas_2(lon_m,lon_M,lat_m)
        elif not lonBool and latBool:
            print 'Cas 3'
            self.Cas_3(lon_m,lat_m,lat_M)
        else:
            print 'Cas4'
            self.Cas_4(lon_m,lon_M,lat_m,lat_M)

            
    def Cas_1(self,lon,lat):
        ''' Ni long ni lat ne croise en bord de la carte
                colle le bon wac sur self.wac '''

        lonf,lonc = self._format_lon(lon)
        latf,latc = self._format_lat(lat)
        wac = 'WAC_GLOBAL_E'+latc+lonc+'_128P.IMG'
        f = os.path.join(self.racine,'PDS_FILE','LROC_WAC',wac)
        self.Wac = BinaryWACTable(f)

    def Cas_2(self,lon_m,lon_M,lat):
        ''' long croise en bord de la carte
        colle le bon wac sur self.wac '''
        
        lonmf,lonm = self._format_lon(lon_m)
        lonMf,lonM = self._format_lon(lon_M)
        latf,latc = self._format_lat(lat)
        
        Wac = BinaryWACTable(os.path.join(self.racine,
                           'PDS_FILE',
                           'LROC_WAC',
                           'WAC_GLOBAL_E'+latc+lonm+'_128P.IMG'))
        Wac_left = Wac
        Wac_Ok = Wac
        X_left_new = Wac_left.X[Wac_left.X > lonmf]
        SizeY_left = Wac_left.Y.shape[0]
        SizeX_left = X_left_new.shape[0]
        mask_left = np.repeat((Wac_left.X[np.newaxis,:] > lonmf),SizeY_left,axis=0)
        Z_left_new = Wac_left.Z[mask_left].reshape((SizeY_left,SizeX_left))
        del Wac,Wac_left
        
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latc+lonM+'_128P.IMG'))
        Wac_right = Wac
        X_right_new = Wac_right.X[Wac_right.X < lonMf]
        SizeY_right = Wac_right.Y.shape[0]
        SizeX_right = X_right_new.shape[0]
        mask_right = np.repeat((Wac_right.X[np.newaxis,:] < lonMf),SizeY_right,axis=0)
        Z_right_new = Wac_right.Z[mask_right].reshape((SizeY_right,SizeX_right))
        del Wac,Wac_right
        
        Wac_Ok.X = np.hstack((X_left_new,X_right_new))
        Wac_Ok.Z = np.hstack((Z_left_new,Z_right_new))

        self.Wac = Wac_Ok

    def Cas_3(self,lon_m,lat_m,lat_M):
        ''' lat croise en bord de la carte
        colle le bon wac sur self.wac '''
        
        loncf,lonc = self._format_lon(lon_m)
        latmf,latm = self._format_lat(lat_m)
        latMf,latM = self._format_lat(lat_M)
        
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latm+lonc+'_128P.IMG'))
        Wac_bot = Wac
        Wac_Ok = Wac
        
        Y_bot_new = Wac_bot.Y[Wac_bot.Y > latmf]
        SizeX_bot = Wac_bot.X.shape[0]
        SizeY_bot = Y_bot_new.shape[0]
        mask_bot = np.repeat((Wac_bot.Y[:,np.newaxis] > latmf),SizeX_bot,axis=1)
        Z_bot_new = Wac_bot.Z[mask_bot].reshape((SizeY_bot,SizeX_bot))
        del Wac,Wac_bot
        
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latM+lonc+'_128P.IMG'))
        Wac_top = Wac
        Y_top_new = Wac_top.Y[Wac_top.Y < latMf]
        SizeX_top = Wac_top.X.shape[0]
        SizeY_top = Y_top_new.shape[0]
        mask_top = np.repeat((Wac_top.Y[:,np.newaxis] < latMf),SizeX_top,axis=1)
        Z_top_new = Wac_top.Z[mask_top].reshape((SizeY_top,SizeX_top))
        del Wac,Wac_top
        
        Wac_Ok.Y = np.hstack((Y_top_new,Y_bot_new))
        Wac_Ok.Z = np.vstack((Z_top_new,Z_bot_new))
        
        self.Wac = Wac_Ok

    def Cas_4(self,lon_m,lon_M,lat_m,lat_M):
        '''Lat crois en bord de la carte et Long asusi
        lon_mLat_M : 00
        lon_m,Lat_m : 10
        lon_M,Lat_M : 01
        lon_M,lat_m : 11 '''
        
        lonmf,lonm = self._format_lon(lon_m)
        lonMf,lonM = self._format_lon(lon_M)
        latmf,latm = self._format_lat(lat_m)
        latMf,latM = self._format_lat(lat_M)

        # Carre haut gauche 00
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latM+lonm+'_128P.IMG'))

        Wac_Ok = Wac
        Wac00 = Wac
        X00 = Wac00.X[Wac00.X>lonmf]
        Y00 = Wac00.Y[Wac00.Y<latMf]
        SizeX = Wac00.X.shape[0]
        SizeY = Wac00.Y.shape[0]
        SizeX00 = X00.shape[0]
        SizeY00 = Y00.shape[0]
        mask00X = np.repeat((Wac00.X[np.newaxis,:] > lonmf),SizeY,axis=0)
        mask00Y = np.repeat((Wac00.Y[:,np.newaxis] < latMf),SizeX,axis=1)
        mask00 = mask00X&mask00Y
        Z00 = Wac00.Z[mask00].reshape((SizeX00,SizeY00))
        del Wac00,Wac
        
        # Carre Haut droit 01
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latM+lonM+'_128P.IMG'))
        Wac01 = Wac
        X01 = Wac01.X[Wac01.X<lonMf]
        Y01 = Wac01.Y[Wac01.Y<latMf]
        SizeX = Wac01.X.shape[0]
        SizeY = Wac01.Y.shape[0]
        SizeX01 = X01.shape[0]
        SizeY01 = Y01.shape[0]
        mask01X = np.repeat((Wac01.X[np.newaxis,:] < lonMf),SizeY,axis=0)
        mask01Y = np.repeat((Wac01.Y[:,np.newaxis] < latMf),SizeX,axis=1)
        mask01 = mask01X&mask01Y
        Z01 = Wac01.Z[mask01].reshape((SizeX01,SizeY01))        
        del Wac01,Wac
        
        # Carre bas gauche 10
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latm+lonm+'_128P.IMG'))
        Wac10 = Wac
        X10 = Wac10.X[Wac10.X>lonmf]
        Y10 = Wac10.Y[Wac10.Y>latmf]
        SizeX = Wac10.X.shape[0]
        SizeY = Wac10.Y.shape[0]
        SizeX10 = X10.shape[0]
        SizeY10 = Y10.shape[0]
        mask10X = np.repeat((Wac10.X[np.newaxis,:] > lonmf),SizeY,axis=0)
        mask10Y = np.repeat((Wac10.Y[:,np.newaxis] > latmf),SizeX,axis=1)
        mask10 = mask10X&mask10Y
        Z10 = Wac10.Z[mask10].reshape((SizeX10,SizeY10))         
        del Wac10,Wac
        
        # Carre bas gauche 11
        Wac = BinaryWACTable(os.path.join(self.racine,
                                          'PDS_FILE',
                                          'LROC_WAC',
                                          'WAC_GLOBAL_E'+latm+lonM+'_128P.IMG'))
        Wac11 = Wac
        X11 = Wac11.X[Wac11.X<lonMf]
        Y11 = Wac11.Y[Wac11.Y>latmf]
        SizeX = Wac11.X.shape[0]
        SizeY = Wac11.Y.shape[0]
        SizeX11 = X11.shape[0]
        SizeY11 = Y11.shape[0]
        mask11X = np.repeat((Wac11.X[np.newaxis,:] < lonMf),SizeY,axis=0)
        mask11Y = np.repeat((Wac11.Y[:,np.newaxis] > latmf),SizeX,axis=1)
        mask11 = mask11X&mask11Y
        Z11 = Wac11.Z[mask11].reshape((SizeX11,SizeY11))
        del Wac11,Wac

        # On rassemble tous
        Wac_Ok.X = np.hstack((X00,X01))
        Wac_Ok.Y = np.hstack((Y00,Y10))
        Z_top = np.hstack((Z00,Z01))
        Z_bot = np.hstack((Z10,Z11))
        Wac_Ok.Z = np.vstack((Z_top,Z_bot))

        self.Wac = Wac_Ok        
        
    def Load_Wac(self):
            
        lon_m,lon_M,lat_m,lat_M = self.Cylindrical_Window(self.Taille_Window,self.Lat,self.Long)
        self.Define_Case(lon_m,lon_M,lat_m,lat_M)

    def Load_Lola(self,name):
            
        f = os.path.join(self.racine,'PDS_FILE','Lola',name)
        self.Lola = BinaryLolaTable(f)

    def Load_Grail(self,name):
            
        f = os.path.join(self.racine,'PDS_FILE','Grail',name)
        self.Grail = BinaryGrailTable(f)

        
    def Crater_Data(self):
        Racine = os.path.join(self.racine,'Data')
        data = Data(64,Racine,'_2')
        df = pd.DataFrame(np.hstack((data.Name,data.Index,data.Lat,data.Long,data.Diameter))
                          ,columns = ['Name','Index','Lat','Long','Diameter'])
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

    def save_fig(self,Name,fig):

        root = '/Users/thorey/Documents/These/Projet/FFC/Classification/Data/Image'
        path = os.path.join(root,Name)
        fig.savefig(path,rasterized=True, dpi=50,bbox_inches='tight',pad_inches=0.1)

    def LROC_Image_Feature(self):
        if self.Wac =='':
            self.Load_Wac()
        
        fig = plt.figure(figsize=(14,14))
        ax1 = fig.add_subplot(111)
        ax1.set_rasterization_zorder(1)
        X,Y,Z = self.Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = self.Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolormesh(X,Y,Z,cmap = cm.gray ,ax  = ax1,zorder =-1)

        return fig
        
    def plot_LROC(self):
        if self.Wac =='':
            self.Load_Wac()
        
        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(111)
        ax1.set_rasterization_zorder(1)
        X,Y,Z = self.Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = self.Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolormesh(X,Y,Z,cmap = cm.gray ,ax  = ax1,zorder =-1)
        
        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=100,marker ='v',zorder =2)

        return fig

        
    def plot_LOLA(self):

        if self.Wac =='' or self.Lola == '':
            print 'load a wac and lola FIRSTTTT'
            sys.exit()
        
        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(111)
        ax1.set_rasterization_zorder(1)
        X,Y,Z = self.Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = self.Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolormesh(X,Y,Z,cmap = cm.gray ,ax  = ax1,zorder =-1)
        Xl,Yl,Zl = self.Lola.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        Xl,Yl = m(Xl,Yl)
        # m.pcolor(Xl,Yl,Zl,cmap='gist_earth', alpha = 0.3 ,zorder=-1)
        m.contourf(Xl,Yl,Zl,100,cmap='gist_earth', alpha = 0.3 , zorder=-1,antialiased=True)
        # cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        # cb.set_label('Topography', size = 24)
        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=100,marker ='v',zorder =2)

        lol,loM,lam,laM = self.Wac.Lambert_Window(0.6*self.Diameter,self.Lat,self.Long)
        m.drawmapscale(loM,lam, self.Long,self.Lat,10,
                       barstyle='fancy', units='km',
                       fontsize=24, yoffset=None,
                       labelstyle='simple',
                       fontcolor='k',
                       fillcolor1='w',
                       fillcolor2='k', ax=ax1,
                       format='%d',
                       zorder=-1)
        ax1.set_title('Crater %s, %d km in diameter'%(self.Name,self.Diameter),size = 24)
        Namefig = self.Name+'_'+str(self.Diameter)+'_Lola.eps'
        
        self.save_fig(Namefig,fig)

        return fig


    def plot_GRAIL(self):

    
        if self.Wac =='' or self.Grail == '':
            print 'load a wac and lola FIRSTTTT'
            sys.exit()
            
        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(111)
        ax1.set_rasterization_zorder(1)
        X,Y,Z = self.Wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        lon_m,lon_M,lat_m,lat_M = self.Wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolormesh(X,Y,Z,cmap = cm.gray ,ax  = ax1,zorder=-1)
        
        Xg,Yg,Zg = self.Grail.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        clevs = np.arange(-50,50,5)
        Xg,Yg = m(Xg,Yg)
        # CS2 = m.contour(Xg,Yg,Zg,clevs,linewidths=0.5,colors='k',antialiased = True ,zorder= -1, )
        CS3 = m.contourf(Xg,Yg,Zg,clevs,cmap=plt.cm.RdBu_r,antialiased = True , alpha = 0.5,zorder =-1)
        cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        cb.set_label('Gravity anomaly',size = 24)

        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=100,marker ='v',zorder =2)

        lol,loM,lam,laM = self.Wac.Lambert_Window(0.6*self.Diameter,self.Lat,self.Long)
        m.drawmapscale(loM,lam, self.Long,self.Lat,10,
                       barstyle='fancy', units='km',
                       fontsize=24, yoffset=None,
                       labelstyle='simple',
                       fontcolor='k',
                       fillcolor1='w',
                       fillcolor2='k', ax=ax1,
                       format='%d',
                       zorder=-1)
        ax1.set_title('Dome %s, %d km in diameter'%(self.Name,self.Diameter),size = 24)
        Namefig = self.Name+'_'+str(self.Diameter)+'_Grail.eps'
        
        self.save_fig(Namefig,fig)

    def Deg(self,radius):
        return radius*360/(2*np.pi*1734.4)

        
        




        


