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

def plot_xyz(lat_lon_v_list, box_size = 6, cmap_name = 'Purples', title = '', path=None):
	"""
	Plots the xyz columns of a database with some value represented as well.

	Args:
		lat_lon_v_list: List of the format [(lat, lng, v), ...]
		box_size: How much coarse graining to do. This is the number of degrees
			of latitude and longitude to average over.
		cmap_name = 'Purples': The matplotlib colormap to use
		title = '': Title for plot
		path = None: If a string is given, the plot is saved to that location. Otherwise
		just displays to screen.


	Returns:
		0: Plot object
	"""
	# Do the coarse graining
	number_of_lons = int(360/box_size)
	number_of_lats = int(180/box_size)
	lat_space = np.linspace(-90,90,number_of_lats+1)
	lon_space = np.linspace(0,360,number_of_lons+1)

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
	z = z_sum/z_number
	# Deal with places with no data
	z = np.ma.masked_invalid(z)
	# Time to plot
	m = Basemap(projection='robin', lon_0 = 0)
	m.drawcoastlines()
	m.drawcountries()

	palette = copy.copy(plt.cm.get_cmap(name=cmap_name))
	palette.set_bad('#c2c3c4', 1.0)

	cs = m.pcolormesh(x, y, z, latlon=True, cmap=palette)

	cbar = m.colorbar(cs, location='bottom', pad="5%")
	cbar.set_label('(as a portion of active substances)')

	plt.title(title)

	plt.show()




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