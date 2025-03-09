import matplotlib.pyplot as plt
import rioxarray
import geopandas
import pandas as pd
import pathlib
import matplotlib.colors as colors



from shapely.geometry import Point
from matplotlib_scalebar.scalebar import ScaleBar
# Read the background
shipping_path = '../../2021_st_All_avg.tif'

borders_EEZ = r"../../eez_boundaries_v10_BE_epsg4326.shp" 

borders_df = geopandas.read_file(borders_EEZ)
shipping_xr = rioxarray.open_rasterio(shipping_path)
shipping_xr = shipping_xr.rio.set_crs('EPSG:3035')

# Read the station coordinates
output_folder = pathlib.Path(r'../../stations_and_planning')
locations_path = output_folder.joinpath('Locations_ETN2.csv')

df = pd.read_csv(locations_path)
df = df.loc[df.stationName.isin(['Gardencity', 'Grafton'])]
# df = df.loc[~df.stationName.isin(['Birkenfels', 'Fairplay', 'Reefballs Belwind'])]
geodf = geopandas.GeoDataFrame(df,
                               geometry=geopandas.points_from_xy(x=df['deployLong'], y=df['deployLat']),
                               crs='epsg:4326')

points = geopandas.GeoSeries(
    [Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=4326
)  # Geographic WGS 84 - degrees
points = points.to_crs(32619)  # Projected WGS 84 - meters
distance_meters = points[0].distance(points[1])


# Read the station coordinates
img_output = pathlib.Path('')

crs = 'EPSG:4326'
minx, miny, maxx, maxy = borders_df.to_crs(crs).total_bounds

# Distribution of all the collected data
# Distribution of all the collected data
fig, ax = plt.subplots()
cmap = 'Blues'

# Adjusting the colorbar position to the left with custom padding
shipping_xr.rio.reproject(crs).rio.clip_box(minx, miny,
                                            maxx, maxy).plot(
    ax=ax, 
    add_colorbar=True, 
    cbar_kwargs={
        'label': r'Vessel density [$h/km^2/month$]', 
        'orientation': 'vertical', 
        'location': 'right', 
        'shrink': 0.8, # Shrink the colorbar to fit within the figure
        'aspect': 20 # Adjust the aspect ratio of the colorbar
    },
    cmap=cmap, 
    zorder=0,
    norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=1, vmax=100, base=10), 
    shading='auto', 
    alpha=0.7
)

borders_df.to_crs(crs).plot(ax=ax, color='k', zorder=1)
geodf.to_crs('epsg:4326').plot('samplingStrategy', ax=ax, markersize=20,  marker='x', color='k', zorder=3)
for x, y, label in zip(geodf.geometry.x, geodf.geometry.y, geodf.stationName):
    ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
ax.add_artist(ScaleBar(distance_meters, location="lower right"))

save_path = r'3_3_figures\data_collection\shipping_map.png'


plt.xlabel('Longitude [degrees East]')
plt.ylabel('Latitude [degrees North]')
plt.title(None)
plt.savefig(save_path, dpi=350, bbox_inches="tight")
# plt.show()