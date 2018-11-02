"""
The main program for execution.

Project: EcstasyData
Path: root/main.py
"""
import get_data
import visualizer as vis
import classifications as clss
import som

import sqlite3

import os

import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import datetime

db_path = 'EcstasyData.sqlite'

# db = get_data.EDataDB('Location.sqlite')
# db.new_copy('EcstasyData.sqlite')
# db.open()
# db.initialize()
# db.load_url('https://www.ecstasydata.org/search.php')
# db.dump_unknown_substances('unknown_substances.txt')
# db.dump_bad_locations('bad_locations.txt')
# db.commit()
# db.close()

# get_data.create_database('https://www.ecstasydata.org/search.php?source=1&Max=5000&style=data_only', db_path)

############# PLOT CONTENT MAP #################
def plot_content(in_path, content_string, out_path=None, title='', box_size=6, mode='average', \
	llcrnrlat=-58, llcrnrlon=-180, urcrnrlat=75, urcrnrlon=180, logplot=False, \
	z_min=0, z_max=1, cmap_name = 'Purples', default_range=False, cb_label='(as a portion of active substances)'):
	"""
	Plots the content_string on a map.

	Args:
		in_path: Path to the database.
		out_path: Path to the image
		content_string: SQL string for the value column of the plot from Content_Map. E.g. '(MDMA_Content + Enactogen_Content) as Content'
	"""

	conn = sqlite3.connect(in_path)
	c = conn.cursor()

	# Plot the ecstasy map first
	c.execute('SELECT Latitude, Longitude, ' + content_string + ' FROM Content_Map;')
	content_data = c.fetchall()

	vis.plot_latlng(content_data, box_size = box_size, title = title, path=out_path, mode=mode, \
		llcrnrlat=llcrnrlat, llcrnrlon=llcrnrlon, urcrnrlat=urcrnrlat, urcrnrlon=urcrnrlon, logplot=logplot, \
		z_min=z_min, z_max=z_max, cmap_name = cmap_name, default_range=default_range, cb_label=cb_label)

	conn.close()

def plot_by_ID(in_path, IDs, out_path=None, title='', box_size=6, mode='average', \
	llcrnrlat=-58, llcrnrlon=-180, urcrnrlat=75, urcrnrlon=180, logplot=False, \
	z_min=0, z_max=1, cmap_name = 'Purples', default_range=False, cb_label='(as a portion of active substances)'):
	"""
	Plots the content_string on a map.

	Args:
		in_path: Path to the database.
		out_path: Path to the image
		content_string: SQL string for the value column of the plot from Content_Map. E.g. '(MDMA_Content + Enactogen_Content) as Content'
	"""

	conn = sqlite3.connect(in_path)
	c = conn.cursor()
	# Plot the ecstasy map first
	c.execute("DROP TABLE IF EXISTS Cluster_Placeholder;")
	c.execute("""
			CREATE TABLE Cluster_Placeholder(
				Pill_ID INTEGER
			);
		""")
	for ID in IDs:
		c.execute("""
				INSERT INTO
					Cluster_Placeholder
				VALUES
					(?);
			""", (ID,))
	c.execute("""
			SELECT
				Location.Latitude,
				Location.Longitude,
				1
			FROM
				Cluster_Placeholder, Location, Pill_Misc
			WHERE
				Cluster_Placeholder.Pill_ID = Pill_Misc.Pill_ID
			AND
				Pill_Misc.Location_ID = Location.Location_ID;
		""")
	content_data = c.fetchall()
	c.execute("DROP TABLE Cluster_Placeholder;")
	conn.commit()
	conn.close()

	vis.plot_latlng(content_data, box_size = box_size, title = title, path=out_path, mode=mode, \
		llcrnrlat=llcrnrlat, llcrnrlon=llcrnrlon, urcrnrlat=urcrnrlat, urcrnrlon=urcrnrlon, logplot=logplot, \
		z_min=z_min, z_max=z_max, cmap_name = cmap_name, default_range=default_range, cb_label=cb_label)

# # Plot world
# plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
# 	out_path='Content_Plots\\Samples.png', title='Number of Samples Tested', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', logplot=True)
# plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA.png', title='MDMA Content')
# plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen.png', title='Enactogen Content')
# plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedelic.png', title='Psychedelic Content')
# plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid.png', title='Cannabinoid Content')
# plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative.png', title='Dissociative Content')
# plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Stimulant.png', title='Stimulant Content')
# plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Depressant.png', title='Depressant Content')
# plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other.png', title='Other Content')

# # Plot North America
# plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
# 	out_path='Content_Plots\\Samples_NA.png', title='Number of Samples Tested (North America)', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', \
# 	llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2, logplot=True)
# plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA_NA.png', title='MDMA Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen_NA.png', title='Enactogen Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedleic_NA.png', title='Psychedelic Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid_NA.png', title='Cannabinoid Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative_NA.png', title='Dissociative Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Stimulant_NA.png', title='Stimulant Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Depressant_NA.png', title='Depressant Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
# plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other_NA.png', title='Other Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)


# # Plot Europe
# plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
# 	out_path='Content_Plots\\Samples_EU.png', title='Number of Samples Tested (Europe)', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', \
# 	llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2, logplot=True)
# plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA_EU.png', title='MDMA Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen_EU.png', title='Enactogen Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedleic_EU.png', title='Psychedelic Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid_EU.png', title='Cannabinoid Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative_EU.png', title='Dissociative Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Stimulant_EU.png', title='Stimulant Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Depressant_EU.png', title='Depressant Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
# plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other_EU.png', title='Other Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)

# # Plot UK
# plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
# 	out_path='Content_Plots\\Samples_UK.png', title='Number of Samples Tested (United Kingdom)', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', \
# 	llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5, logplot=True)
# plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA_UK.png', title='MDMA Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen_UK.png', title='Enactogen Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedleic_UK.png', title='Psychedelic Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid_UK.png', title='Cannabinoid Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative_UK.png', title='Dissociative Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Stimulant_UK.png', title='Stimulant Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Depressant_UK.png', title='Depressant Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
# plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other_UK.png', title='Other Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)

################### SOM ANALYSIS ############################
def build_SOM(db_path, query, N, path = None, random_seed=21893698, dimensions = None, \
	normalization_mode = 'None', ID = False):
	# Create a som
	with get_data.EDataDB(db_path) as db:
		db.open()
		db.c.execute(query)
		data = db.c.fetchall()
		print('Analysing %d points...' % (len(data), ))
		somap = som.ClassifiedSOM(data, random_seed=random_seed,
			dimensions = dimensions, normalization_mode = normalization_mode,
			ID = ID, sigma = 0.8)

	# Initialize weight from data
	somap.random_weights_init(somap.data)

	# Train SOM
	somap.train(N)

	# Generate informative maps
	somap.generate_distance_map()
	somap.generate_activation_response()

	# Save SOM
	if path != None:
		with open(path, 'wb') as out_f:
			pickle.dump(somap, out_f)
	return somap

# Queries for SOM data IN US, EU, and UK
query_US = """
		SELECT
			Pill_Misc.Date,
		    Location.Latitude,
		    Location.Longitude,
		    SOM_Classification.MDMA_Content,
		    SOM_Classification.Enactogen_Content,
		    SOM_Classification.Psychedelic_Content,
		    SOM_Classification.Cannabinoid_Content,
		    SOM_Classification.Dissociative_Content,
		    SOM_Classification.Stimulant_Content,
		    SOM_Classification.Depressant_Content,
		    SOM_Classification.Other_Content
		FROM
			Pill_Misc, Location, SOM_Classification
		WHERE
			Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
		AND
			Pill_Misc.Location_ID = Location.Location_ID
		AND
			date(Pill_Misc.Date) BETWEEN date('2008-01-01') AND date('2019-01-01')
		AND 
			Location.Latitude BETWEEN 15 AND 60
		AND
			Location.Longitude BETWEEN -135 AND -60;
		"""

query_no_coords_NA = """
		SELECT
			Pill_Misc.Pill_ID,
			Pill_Misc.Date,
		    SOM_Classification.MDMA_Content,
		    SOM_Classification.Enactogen_Content,
		    SOM_Classification.Psychedelic_Content,
		    SOM_Classification.Cannabinoid_Content,
		    SOM_Classification.Dissociative_Content,
		    SOM_Classification.Stimulant_Content,
		    SOM_Classification.Depressant_Content,
		    SOM_Classification.Other_Content
		FROM
			Pill_Misc, SOM_Classification, Location
		WHERE
			Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
		AND
			Pill_Misc.Location_ID = Location.Location_ID
		AND
			date(Pill_Misc.Date) BETWEEN date('2008-01-01') AND date('2019-01-01')
		AND 
			Location.Latitude BETWEEN 15 AND 60
		AND
			Location.Longitude BETWEEN -135 AND -60
		AND SOM_Classification.MDMA_Content < 1;
	"""

query_no_coords = """
		SELECT
			Pill_Misc.Pill_ID,
			Pill_Misc.Date,
		    SOM_Classification.MDMA_Content,
		    SOM_Classification.Enactogen_Content,
		    SOM_Classification.Psychedelic_Content,
		    SOM_Classification.Cannabinoid_Content,
		    SOM_Classification.Dissociative_Content,
		    SOM_Classification.Stimulant_Content,
		    SOM_Classification.Depressant_Content,
		    SOM_Classification.Other_Content
		FROM
			Pill_Misc, SOM_Classification, Location
		WHERE
			Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
		AND
			Pill_Misc.Location_ID = Location.Location_ID
		AND
			date(Pill_Misc.Date) BETWEEN date('2008-01-01') AND date('2019-01-01');
	"""

query_no_coords_EU = """
		SELECT
			Pill_Misc.Pill_ID,
			Pill_Misc.Date,
		    SOM_Classification.MDMA_Content,
		    SOM_Classification.Enactogen_Content,
		    SOM_Classification.Psychedelic_Content,
		    SOM_Classification.Cannabinoid_Content,
		    SOM_Classification.Dissociative_Content,
		    SOM_Classification.Stimulant_Content,
		    SOM_Classification.Depressant_Content,
		    SOM_Classification.Other_Content
		FROM
			Pill_Misc, SOM_Classification, Location
		WHERE
			Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
		AND
			Pill_Misc.Location_ID = Location.Location_ID
		AND
			date(Pill_Misc.Date) BETWEEN date('2008-01-01') AND date('2019-01-01')
		AND 
			Location.Latitude BETWEEN 35 AND 65
		AND
			Location.Longitude BETWEEN -15 AND 40
		AND
			SOM_Classification.MDMA_Content < 1;
	"""

query_EU = """
		SELECT
			Pill_Misc.Date,
		    Location.Latitude,
		    Location.Longitude,
		    SOM_Classification.MDMA_Content,
		    SOM_Classification.Enactogen_Content,
		    SOM_Classification.Psychedelic_Content,
		    SOM_Classification.Cannabinoid_Content,
		    SOM_Classification.Dissociative_Content,
		    SOM_Classification.Stimulant_Content,
		    SOM_Classification.Depressant_Content,
		    SOM_Classification.Other_Content
		FROM
			Pill_Misc, Location, SOM_Classification
		WHERE
			Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
		AND
			Pill_Misc.Location_ID = Location.Location_ID
		AND
			date(Pill_Misc.Date) BETWEEN date('2008-01-01') AND date('2019-01-01')
		AND 
			Location.Latitude BETWEEN 35 AND 65
		AND
			Location.Longitude BETWEEN -15 AND 40;
		"""

query_UK = """
		SELECT
			Pill_Misc.Date,
		    Location.Latitude,
		    Location.Longitude,
		    SOM_Classification.MDMA_Content,
		    SOM_Classification.Enactogen_Content,
		    SOM_Classification.Psychedelic_Content,
		    SOM_Classification.Cannabinoid_Content,
		    SOM_Classification.Dissociative_Content,
		    SOM_Classification.Stimulant_Content,
		    SOM_Classification.Depressant_Content,
		    SOM_Classification.Other_Content
		FROM
			Pill_Misc, Location, SOM_Classification
		WHERE
			Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
		AND
			Pill_Misc.Location_ID = Location.Location_ID
		AND
			date(Pill_Misc.Date) BETWEEN date('2008-01-01') AND date('2019-01-01')
		AND 
			Location.Latitude BETWEEN 49.5 AND 59
		AND
			Location.Longitude BETWEEN -12 AND 44;
		"""

########### Build and train SOM ################
num_iter  = 100000
shape = (15,15)
normalization_mode = 'None'
directory = 'NoCoordinates/%sNormalized/NA' % (normalization_mode,)

somap = build_SOM(db_path, query_no_coords_NA,
		N = num_iter,
		path = 'pickles/%s/SOM_2008-2018_%d_%s.pickle' % (directory, num_iter, str(shape)),
		dimensions = shape,
		normalization_mode = normalization_mode,
		ID = True)

# with open('pickles/%s/SOM_2008-2018_%d_%s.pickle' % (directory, num_iter, str(somap.shape)), 'wb') as out_f:
	# pickle.dump(somap, out_f)

########### ALTER PICKLE ################

# with open('pickles/GaussianNormalized/US/SOM_2008-2018_nonpure_%d_%s.pickle' % (num_iter, str(shape)), 'rb') as in_f:
# 	somap = pickle.load(in_f)

# DO SOMETHING

# with open('pickles/GaussianNormalized/SOM_2008-2018_nonpure_%d_%s.pickle' % (num_iter, str(shape)), 'wb') as out_f:
# 	pickle.dump(somap, out_f)
# exit()

################ CLUSTERING ######################
with open('pickles/%s/SOM_2008-2018_%d_%s.pickle' % (directory, num_iter, str(shape)), 'rb') as in_f:
	somap = pickle.load(in_f)

print(len(somap.data))

somap.plot_activation_response(path='SOM_Plots/%s/ActivationResponse_2008-2018_%d_%s.png' % (directory, num_iter, str(somap.shape)))
somap.plot_distance_map(path = 'SOM_Plots/%s/DistanceMap_2008-2018_%d_%s.png' % (directory, num_iter, str(somap.shape)))

somap.cluster(0.08)
somap.cluster_analysis()

somap.plot_clusters()
somap.plot_clusters(normalization = 'size', path = 'SOM_Plots/%s/ClustersNormalized_2008-2018_%d_%s.png' % (directory, num_iter, str(shape)))
somap.plot_clusters(path = 'SOM_Plots/%s/Clusters_2008-2018_%d_%s.png' % (directory, num_iter, str(shape)))
somap.cluster_report('SOM_Plots/%s/ClusterReport_2008-2018_%d_%s.txt' % (directory, num_iter, str(shape)))

# Plotting on Map
for i, IDs in enumerate(somap.member_IDs(x) for x in somap.clusters):
	plot_by_ID(db_path, IDs,
		out_path='SOM_Plots/%s/Cluster%d_2008-2018_%d_%s.png' % (directory, i, num_iter, str(shape)),
		title='Cluster %d' % (i,), mode = 'sum', z_min = 1, z_max = None, logplot = False,
		llcrnrlat=23, llcrnrlon=-135, urcrnrlat=53, urcrnrlon=-60, box_size=2)

vis.plot_clusters(somap.cluster_analysis()[:8], somap.denormalize,
	llcrnrlat=23, llcrnrlon=-135, urcrnrlat=52, urcrnrlon=-60,
	title = 'Clusters for 2008-2018_nonpure_%d_%s.png' % (num_iter, str(shape)),
	path = 'SOM_Plots/%s/ClusterReport_2008-2018_nonpure_%d_%s.png' % (directory, num_iter, str(shape)))



