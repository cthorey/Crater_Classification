import os
from Data_utils import *
from Extract_Array import *

platform = 'laptop'

if platform == 'clavius':
    racine = '/Users/clement/Classification/'
elif platform == 'laptop':
    racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/'

# Fichier de base !
dat = os.path.join(racine,'Data')
df = Data(64,dat,'_3')

data = df.To_DetermineDF()

for i,row in data.iterrows():
    crater = Crater(row.Index,'i',racine)
    crater.plot_LOLA(True)
