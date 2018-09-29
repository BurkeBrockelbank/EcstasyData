"""
This program fetches data from ecstacy data.org

Project: EcstasyData
Path: root/visualizer.py
"""
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap, cm
import bisect

def lat_lng_to_x_y_z(latitude, longitude, radians = False):
	"""
	Converts latitude and longitude to a (x,y,z) tuple.
	Args:
		latitude: Latitude in degrees.
		longitude: Longitude in degrees.
		radians = False: True if latitude and longitude are given in radians

	Returns:
		0: (x,y,z)
	"""
	if radians:
		lat = latitude
		lng = longitude
	else:
		lat = latitude/180*np.pi
		lng = longitude/180*np.pi
	x = np.cos(lng)*np.cos(lat)
	y = np.sin(lng)*np.cos(lat)
	z = np.sin(lat)
	return (x,y,z)

def x_y_z_to_lat_lng(x, y, z, radians = False):
	"""
	Convertes x, y, and z values into a latitude longitude tuple.

	Args:
		x:
		y:
		z:
		radians = False: True if latitude and longitude are given in radians

	Returns:
		0: (latitude, longitude)
	"""
	lat = np.arcsin(z)
	lng = np.arctan(y/x)


	if not radians:
		lat = lat*180/np.pi
		lng = lng*180/np.pi
	return (lat, lng)

def plot_latlng(lat_lon_v_list, box_size = 6, cmap_name = 'Purples', title = '', path=None, \
	z_min=0, z_max=1, llcrnrlat=-58, llcrnrlon=-180, urcrnrlat=75, urcrnrlon=180, mode='average',
	default_range=False, cb_label='(as a portion of active substances)', logplot = False):
	"""
	Plots the latitude and longitude columns of a database with some value represented as well.

	Args:
		lat_lon_v_list: List of the format [(lat, lng, v), ...]
		box_size: How much coarse graining to do. This is the number of degrees
			of latitude and longitude to average over.
		cmap_name = 'Purples': The matplotlib colormap to use
		title = '': Title for plot
		path = None: If a string is given, the plot is saved to that location. Otherwise
		just displays to screen.
		z_min: Default 0. Minimum value for the color scale.
		z_max: Default 1. Maximum value for the color scale.
		lat_0: The latitude of the center of the map
		lon_0: The longitude of the center of the map
		width: The width of the map in the projection units
		height: The height of the map in the projection units

	Returns:
		0: Plot object
	"""
	# Do the coarse graining
	width = urcrnrlon-llcrnrlon
	height = urcrnrlat-llcrnrlat
	number_of_lons = int(width/box_size)
	number_of_lats = int(height/box_size)
	lat_space = np.linspace(llcrnrlat,urcrnrlat,number_of_lats+1)
	lon_space = np.linspace(llcrnrlon,urcrnrlon,number_of_lons+1)

	# Build the x and y grid for lat and lon
	x = np.zeros((number_of_lats+1, number_of_lons+1))
	y = np.zeros((number_of_lats+1, number_of_lons+1))
	for i in range(number_of_lats+1):
		x[i,:] = lon_space
	for j in range(number_of_lons+1):
		y[:,j] = lat_space

	# Build the z grid for values
	z_sum = np.zeros((number_of_lats, number_of_lons))
	z_number = np.zeros((number_of_lats, number_of_lons))
	for lat, lon, v in lat_lon_v_list:
		if lat < llcrnrlat or lat > urcrnrlat or lon < llcrnrlon or lon > urcrnrlon:
			# Out of range:
			continue
		# Find grid spot
		i = bisect.bisect_left(lat_space, lat)-1
		if i == -1:
			i == 0
		j = bisect.bisect_left(lon_space, lon)-1
		if j == -1:
			j = 0
		# Insert data
		z_sum[i,j] += v
		z_number[i,j] += 1
	if mode == 'sum':
		z = z_sum
	elif mode == 'average':
		z = z_sum/z_number
	# Deal with places with no data
	z = np.ma.masked_where(z_number==0, z)

	# Time to plot
	m = Basemap(projection='merc', resolution='l', \
		llcrnrlat=llcrnrlat, llcrnrlon=llcrnrlon, urcrnrlat=urcrnrlat, urcrnrlon=urcrnrlon)
	m.drawcoastlines()
	m.drawcountries()
	m.drawstates()

	palette = copy.copy(plt.cm.get_cmap(name=cmap_name))
	palette.set_bad('#c2c3c4', 1.0)

	if logplot:
		norm = matplotlib.colors.LogNorm()
	else:
		norm = matplotlib.colors.Normalize()

	if default_range:
		cs = m.pcolormesh(x, y, z, latlon=True, cmap=palette, norm=norm)
	else:
		cs = m.pcolormesh(x, y, z, latlon=True, cmap=palette, vmin=z_min,vmax=z_max, norm=norm)

	cbar = m.colorbar(cs, location='bottom', pad="5%")
	cbar.set_label(cb_label)

	plt.title(title)

	if path == None:
		plt.show()
	else:
		plt.savefig(path)
		plt.clf()




# m = Basemap(projection='robin',lon_0=0,resolution='c')

# m.drawcoastlines()
# m.drawcountries()

# rand_data = np.random.rand(1000,3)
# print(rand_data[:,2].mean())
# rand_data[:,0] = (rand_data[:,0]-0.5)*180
# rand_data[:,1] = rand_data[:,1]*360

# plot_xyz(rand_data, title='MDMA Purity (Random data for example)')

# lats = rand_data[:,0]
# lons = rand_data[:, 1]
# times = rand_data[:,2]

# m.scatter(lons,lats,latlon=True)

# plt.show()

# dx, dy = 0.05, 0.05
# y, x = np.mgrid[slice(1, 5 + dy, dy),
#                 slice(1, 5 + dx, dx)]

# z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)

# print(x)
# print()
# print(y)