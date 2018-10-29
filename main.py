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
def build_SOM(db_path, query, N, path = None, random_seed=21893698, dimensions = None):
	# Create a som
	with get_data.EDataDB(db_path) as db:
		db.open()
		db.c.execute(query)
		somap = som.ClassifiedSOM(db.c.fetchall(), random_seed=random_seed, dimensions = dimensions)

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

# # Testing non-pure ecstasy pills
# num_iter  = 100000
# somap = build_SOM(db_path, """
# 		SELECT * FROM SOM_Data
# 		WHERE
# 			MDMA_Content < 1
# 		AND
# 			date(Date) BETWEEN date('2008-01-01') AND date('2019-01-01')
# 		""",
# 		N = num_iter,
# 		path = 'pickles/som_2008-2018_nonpure_1000000.pickle',
# 		dimensions = (100,100))

# somap.plot_distance_map(path='SOM_Plots/ActivationResponse_2008-2018_nonpure_%d_%s.png' % (num_iter, str(somap.shape)))
# somap.plot_activation_response('SOM_Plots/DistanceMap_2008-2018_nonpure_%d_%s.png' % (num_iter, str(somap.shape)))

# with open('pickles/som_2008-2018_nonpure_%d_%s.pickle' % (num_iter, str(somap.shape)), 'wb') as out_f:
# 	pickle.dump(somap, out_f)

with open('pickles/som_2008-2018_nonpure_100000_(100, 100).pickle', 'rb') as in_f:
	somap = pickle.load(in_f)

# somap.clear_clusters()

# with open('pickles/som_2008-2018_nonpure_100000_(1, 1).pickle', 'wb') as out_f:
# 	pickle.dump(somap, out_f)
# exit()
print(len(somap.data))

somap.cluster(0.08)

somap.plot_clusters(normalization = 'size', path = 'SOM_Plots/ClustersNormalized_2008-2018_nonpure_100000_(100, 100).png')
somap.plot_clusters(path = 'SOM_Plots/Clusters_2008-2018_nonpure_100000_(100, 100).png')
somap.cluster_report('cluster_report.txt')

somap.plot_distance_map(path = 'SOM_Plots/ActivationResponse_2008-2018_nonpure_100000_(100, 100).png')
somap.plot_activation_response(path = 'SOM_Plots/DistanceMap_2008-2018_nonpure_100000_(100, 100).png')



