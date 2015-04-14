import sys,os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Load_G(object):

    '''
    The aim of this class is to give a set of tool to extract Dataframe from
    the file in Data_Observations or data_Synthetic !! :)

    
    Class parameter:
    
    Root = Location des fichiers sources, output de data_observation.py
    Range = Range des diametre qu'ont veut sortir
    
    attribut :
    Lola = 'it_Lola_' Correspond au fichier LOLA CHOISI,
    Les differents possible sont it_Lola_ (la base) et
    LDEM_64. (64 pix par degree)
               
    Grav = 'JGGRAIL_900C9A_CrustAnom' Correspond au model CHOISI,
    Les differents possible sont dans le dossier GRAV
             

    method :
    - m_load_C(): Method that take no argument and that return
    a dataframe containing the information on UC
         
    - m_load_F(): Method that take no argument and that return
    a dataframe containing the information on FFC

    - m_all_mask_C: Method that take no argument and that return
    a dict with the key are the subset of the main dataframe (
    obtained with load_C()) according to different locations.
    Locations are defined as followed :
    
    C_ALL : Corrrespond to all the data from Head2010 (complex craters larger than 20 km) (Of course, FFC are remove from this)
    C_SPA : Part of these craters that are within SPA bassin
    C_HIGH : Part of these craters that are outside of the maria
    C_MARIA : Part of these craters in contact or within the maria
    C_H_SPA : Part of these craters outside the maria and outside SPA
    C_C_FFC : Part of these craters that are close from FFCs
    C_H_B_184 : Part of these craters that are away from basin larger than 184 km
    C_H_B_300 : Part of these craters that are away from basin larger than 300 km
    C_H_B_400 : Part of these craters that are away from basin larger than 400 km
    
    For instance, if I ask all_mask_C(self)['C_SPA'], I'll get the subset of F corr
    espondign to crater within SPA and etc... Details on the locations can be found
    in Data_Format.py

    - m_mask_C(mask): Method that take a list with some locations (listed above)
    and that return a subset of the dict all_mask_C. mask is for example
    ['C_ALL','C_SPA']

    - m_all_mask_F(self): Method that take no argument and that return
    a dict with the key are the subset of the main dataframe (
    obtained with load_F()) according to different locations.
    Locations are defined as followed :
    
    F_ALL : Corrrespond to all the FFCs from Jozwiak2012
    F_SPA : Part of these craters that are within SPA bassin
    F_HIGH : Part of these craters that are outside of the maria
    F_MARIA : Part of these craters in contact or within the maria
    F_H_SPA : Part of these craters outside the maria and outside SPA
    F_H_B_184 : Part of these craters that are away from basin larger than 184 km
    F_H_B_300 : Part of these craters that are away from basin larger than 300 km
    F_H_B_400 : Part of these craters that are away from basin larger than 400 km
    For instance, if I ask all_mask_F(self)['F_SPA'], I'll get the subset of F corr
    espondign to crater within SPA and etc... Details on the locations can be found
    in Data_Format.py

    - m_mask_F(mask): Method that take a list with some locations (listed above)
    and that return a subset of the dict all_mask_F. mask is for example
    ['F_ALL','F_SPA']

    - m_all_mask_F_C() : Method that take no argument and that return a merge dict
    containing the dict from all_mask_F and from all_mask_C

    - m_mask_F_C(mask_F,mask_C) takes to arguments, list that contains locations and
    mermge the two output dictionnary from mask_C and mask_F

    - m_all_loc_F() : takes no argument and return the list of possible
    locaiton for FFC

    - m_all_loc_C() : takes no argument and return the list of possible
    locaiton for UC
    
    - m_all_loc_F_C() : takes no argument and return the list of possible
    locaiton for FFC and UC 

    '''
    Root = '/Users/thorey/Documents/These/Projet/FFC/Gravi_GRAIL/Traitement/Data_Observation/'
    Root_F = '/Users/thorey/Documents/These/Projet/FFC/Gravi_GRAIL/Traitement/Data_Observation/'
    Root_S = '/Users/thorey/Documents/These/Projet/FFC/Gravi_GRAIL/Traitement/GRAIL_SYNTH/'
    Range = 20,184

    def __init__(self):
        self.Lola = 'LDEM_64_'
        # self.Lola = 'it_Lola_'
        self.Grav = 'JGGRAIL_900C9A_CrustAnom'

    def m_load_C(self):

        '''Method that take no argument and that return
        a dataframe containing the information on UC'''

        C = pd.read_csv(Load_G.Root+self.Lola+self.Grav+'_Crater_Head_10m.txt')
        return C

    def m_load_F(self):
        
        '''Method that take no argument and that return
        a dataframe containing the information on FFC'''

        F = pd.read_csv(Load_G.Root_F+self.Lola+self.Grav+'_FFC_10m.txt')
        return F

    def m_S_load_F(self):
        
        '''Method that take no argument and that return
        a dataframe containing the information on FFC'''

        F = pd.read_csv(Load_G.Root_S+self.Lola+self.Grav+'_FFC_Lundi.txt')
        F = F[(F.Diameter>min(Load_G.Range)) & (F.Diameter<max(Load_G.Range))]
        if 'S_mean_80' not in F.columns:
            print 'Pas de S_mean_80, on le met efal a S_min'
            F['S_mean_80'] = F.S_min_80
        return F
    
    
    def m_all_mask_C(self):

        ''' Method that take no argument and that return
        a dict with the key are the subset of the main dataframe
        (obtained with load_C()) according to different locations. '''
        
        C = self.m_load_C()
        mask_C = {'C_ALL' : C[C.Mask_Not_FFC],
                  'C_SPA' : C[~C.Mask_SPA_Out & C.Mask_Not_FFC],
                  'C_F_SPA': C[~C.Mask_SPA_Out & C.Mask_Not_FFC & C.Mask_C_FFC],
                  'C_H'   : C[C.Mask_SPA_Out & C.Mask_Highlands & C.Mask_Not_FFC],
                  'C_M'   : C[C.Mask_SPA_Out & ~C.Mask_Highlands & C.Mask_Not_FFC],
                  'C_F_H' : C[C.Mask_SPA_Out & C.Mask_Not_FFC & C.Mask_C_FFC & C.Mask_Highlands],
                  'C_F_M' : C[C.Mask_SPA_Out & C.Mask_Not_FFC & C.Mask_C_FFC & ~C.Mask_Highlands],
                  'C_H_B_184' : C[C.Mask_Basin_Out_184 & C.Mask_Highlands & C.Mask_Not_FFC],
                  'C_H_B_300' : C[C.Mask_Basin_Out_300 & C.Mask_Highlands & C.Mask_Not_FFC],
                  'C_H_B_400' : C[C.Mask_Basin_Out_400 & C.Mask_Highlands & C.Mask_Not_FFC]}
        for elt in np.arange(10,200,10):
            mask_C.update({'C_H_'+str(elt) : C[C.Mask_SPA_Out & C.Mask_Highlands & C.Mask_Not_FFC & C['Mask_Highlands_'+str(elt)]]})
            mask_C.update({'C_M_'+str(elt) : C[C.Mask_SPA_Out & C.Mask_Highlands & C.Mask_Not_FFC & ~C['Mask_Highlands_'+str(elt)]]})
            mask_C.update({'C_F_H_'+str(elt) : C[C.Mask_SPA_Out & C.Mask_Not_FFC & C.Mask_C_FFC & C['Mask_Highlands_'+str(elt)] & C.Mask_Highlands]})
        for elt in np.arange(10,260,50):
            mask_C.update({'C_H_S_'+str(elt)+'_'+str(elt+50) : C[C.Mask_SPA_Out & C.Mask_Not_FFC & C['Mask_Highlands_'+str(elt)] & ~C['Mask_Highlands_'+str(elt+50)] & C.Mask_Highlands]})
                                            
        return mask_C
    
    def m_mask_C(self,mask):
    
        '''Method that take a list with some locations (listed above)
        and that return a subset of the dict all_mask_C'''
    
        mask_C = self.m_all_mask_C()
        d = dict.fromkeys(mask)
        
        return {k : v for k, v in mask_C.iteritems() if k in d.iterkeys()}
    
    def m_all_mask_F(self):

        ''' Method that take no argument and that return
        a dict with the key are the subset of the main dataframe (
        obtained with load_F()) according to different locations.'''
        
        F = self.m_load_F()
        mask_F = {'F_ALL': F[F.Diameter == F.Diameter],
                  'F_SPA': F[~F.Mask_SPA_Out],
                  'F_H'  : F[F.Mask_Highlands & F.Mask_SPA_Out],
                  'F_M'  : F[~F.Mask_Highlands &F.Mask_SPA_Out],
                  'F_H_B_184' : F[F.Mask_Basin_Out_184 & F.Mask_Highlands],
                  'F_H_B_300' : F[F.Mask_Basin_Out_300 & F.Mask_Highlands],
                  'F_H_B_400' : F[F.Mask_Basin_Out_400 & F.Mask_Highlands]}
        for elt in np.arange(10,200,10):
            mask_F.update({'F_H_'+str(elt) : F[F.Mask_SPA_Out & F.Mask_Highlands & F['Mask_Highlands_'+str(elt)]]})
            mask_F.update({'F_M_'+str(elt) : F[F.Mask_SPA_Out & F.Mask_Highlands & ~F['Mask_Highlands_'+str(elt)]]})
        for elt in np.arange(10,260,50):
            mask_F.update({'F_H_S_'+str(elt)+'_'+str(elt+50) : F[F.Mask_SPA_Out & F['Mask_Highlands_'+str(elt)] & ~F['Mask_Highlands_'+str(elt+50)] & F.Mask_Highlands]})
                
        return mask_F

    def m_S_all_mask_F(self):

        ''' Method that take no argument and that return
        a dict with the key are the subset of the main dataframe (
        obtained with load_F()) according to different locations.'''
        
        F = self.m_S_load_F()
        mask_F = {'F_ALL': F[F.Diameter == F.Diameter],
                  'F_SPA': F[~F.Mask_SPA_Out],
                  'F_H'  : F[F.Mask_Highlands & F.Mask_SPA_Out],
                  'F_M'  : F[~F.Mask_Highlands &F.Mask_SPA_Out],
                  'F_H_B_184' : F[F.Mask_Basin_Out_184 & F.Mask_Highlands],
                  'F_H_B_300' : F[F.Mask_Basin_Out_300 & F.Mask_Highlands],
                  'F_H_B_400' : F[F.Mask_Basin_Out_400 & F.Mask_Highlands]}
        #for elt in np.arange(10,900,10):
        #    mask_F.update({'F_H_'+str(elt) : F[F.Mask_SPA_Out & F.Mask_Highlands & F['Mask_Highlands_'+str(elt)]]})
        #    mask_F.update({'F_M_'+str(elt) : F[F.Mask_SPA_Out & F.Mask_Highlands & ~F['Mask_Highlands_'+str(elt)]]})
        #for elt in np.arange(10,260,50):
        #    mask_F.update({'F_H_S_'+str(elt)+'_'+str(elt+50) : F[F.Mask_SPA_Out & F['Mask_Highlands_'+str(elt)] & ~F['Mask_Highlands_'+str(elt+50)] & F.Mask_Highlands]})
                
        return mask_F

    def m_S_mask_F(self,mask):

        '''Method that take a list with some locations (listed above)
        and that return a subset of the dict all_mask_F'''

        mask_F = self.m_S_all_mask_F()
        d = dict.fromkeys(mask)
        return {k : v for k, v in mask_F.iteritems() if k in d.iterkeys()}

    def m_mask_F(self,mask):

        '''Method that take a list with some locations (listed above)
        and that return a subset of the dict all_mask_F'''

        mask_F = self.m_all_mask_F()
        d = dict.fromkeys(mask)
        return {k : v for k, v in mask_F.iteritems() if k in d.iterkeys()}
    
    def m_all_mask_F_C(self):

        '''Method that take no argument and that return a merge dict
        containing the dict from all_mask_F and from all_mask_C'''
        
        mask_C = self.m_all_mask_C() 
        mask_F = self.m_all_mask_F()
        mask = dict(mask_C.items()+mask_F.items())
        return mask

    def m_mask_F_C(self,mask_F,mask_C):

        '''takes to arguments, list that contains locations and
        merge the two output dictionnary from mask_C and mask_F'''
        
        mask_C = self.m_mask_C(mask_C) 
        mask_F = self.m_mask_F(mask_F)
        mask = dict(mask_C.items()+mask_F.items())
        return mask

    def m_all_loc_F(self):

        ''' takes no argument and return the list of possible
        locaiton for FFC''' 

        mask = self.m_all_mask_F()
        return mask.keys()

    def m_all_loc_C(self):

        ''' takes no argument and return the list of possible
        locaiton for UC''' 

        mask = self.m_all_mask_C()
        return mask.keys()

    def m_all_loc_C_F(self):

        ''' takes no argument and return the list of possible
        locaiton for UC and FFC''' 

        mask = self.m_all_mask_F_C()
        return mask.keys()
