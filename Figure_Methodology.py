from cartopy.io import shapereader
import cartopy.crs as ccrs
from matplotlib import cm
import matplotlib.pylab as plt
import os
from Extract_Array import *

racine = '/Users/thorey/Documents/These/Projet/FFC/Classification/'
#shp = shapereader.Reader(os.path.join(racine,'PDS_FILE','LROC_GLOBAL_MARE','LROC_GLOBAL_MARE'))

fig = plt.figure(figsize=(20, 17))

ax = plt.subplot(1,1,1, projection=ccrs.Orthographic(0.0,0.0))
ax.set_global()
    
base = BinaryWACTable(os.path.join(racine,'PDS_FILE','LROC_WAC','WAC_GLOBAL_E000N1800_004P.IMG'))
lons,lats = np.meshgrid(base.X,base.Y)
ax.pcolormesh( lons , lats , base.Z , cmap=cm.gray,
              vmin = base.Z.min(),
              vmax = base.Z.max(),
              transform = ccrs.PlateCarree())

base = BinaryGrailTable(os.path.join(racine,'PDS_FILE','Grail','34_12_3220_900_80_misfit_rad'))
#base = BinaryLolaTable(os.path.join(racine,'PDS_FILE','Lola','ldem_4'))
lons,lats = np.meshgrid(base.X,base.Y)
ax.contourf( lons , lats , base.Z , 50,
          cmap='jet', 
          transform = ccrs.PlateCarree(),
           alpha = 0.5,
           antialiased = True)

#ax.background_patch.set_visible(False)
#ax.outline_patch.set_visible(False)
#for record, Maria in zip(shp.records(), shp.geometries()):
#    ax.add_geometries([Maria], ccrs.PlateCarree(),
#                      facecolor='k', edgecolor='black',zorder=0)


image = '/Users/thorey/Documents/These/Projet/FFC/Classification/Data/Image'
path = os.path.join(image,'Methodo.png')
fig.savefig(path,rasterized=True, dpi=100,bbox_inches='tight',pad_inches=0.1)
