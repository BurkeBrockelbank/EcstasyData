"""
The main program for execution.

Project: EcstasyData
Path: root/main.py
"""
import get_data
import visualizer as vis
import classifications as clss

import sqlite3

import os

db_path = 'EcstasyData.sqlite'

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
plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
	out_path='Content_Plots\\Samples.png', title='Number of Samples Tested', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', logplot=True)
plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA.png', title='MDMA Content')
plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen.png', title='Enactogen Content')
plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedelic.png', title='Psychedelic Content')
plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid.png', title='Cannabinoid Content')
plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative.png', title='Dissociative Content')
plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Stimulant.png', title='Stimulant Content')
plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Depressant.png', title='Depressant Content')
plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other.png', title='Other Content')

# # Plot North America
plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
	out_path='Content_Plots\\Samples_NA.png', title='Number of Samples Tested (North America)', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', \
	llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2, logplot=True)
plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA_NA.png', title='MDMA Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen_NA.png', title='Enactogen Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedleic_NA.png', title='Psychedelic Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid_NA.png', title='Cannabinoid Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative_NA.png', title='Dissociative Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Dissociative_NA.png', title='Stimulant Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Dissociative_NA.png', title='Depressant Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)
plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other_NA.png', title='Other Content (North America)', llcrnrlat=15, llcrnrlon=-135, urcrnrlat=60, urcrnrlon=-60, box_size=2)


# # Plot Europe
plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
	out_path='Content_Plots\\Samples_EU.png', title='Number of Samples Tested (Europe)', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', \
	llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2, logplot=True)
plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA_EU.png', title='MDMA Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen_EU.png', title='Enactogen Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedleic_EU.png', title='Psychedelic Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid_EU.png', title='Cannabinoid Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative_EU.png', title='Dissociative Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Dissociative_EU.png', title='Stimulant Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Dissociative_EU.png', title='Depressant Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)
plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other_EU.png', title='Other Content (Europe)', llcrnrlat=35, llcrnrlon=-15, urcrnrlat=65, urcrnrlon=40, box_size=2)

# Plot UK
plot_content(db_path, '(MDMA_Content+Enactogen_Content+Psychedelic_Content+Cannabinoid_Content+Dissociative_Content+Stimulant_Content+Depressant_Content+Other_Content) as Content', \
	out_path='Content_Plots\\Samples_UK.png', title='Number of Samples Tested (United Kingdom)', mode='sum', z_min=1, z_max=600, cmap_name='Reds', cb_label='', \
	llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5, logplot=True)
plot_content(db_path, 'MDMA_Content as Content', out_path='Content_Plots\\MDMA_UK.png', title='MDMA Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Enactogen_Content as Content', out_path='Content_Plots\\Enactogen_UK.png', title='Enactogen Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Psychedelic_Content as Content', out_path='Content_Plots\\Psychedleic_UK.png', title='Psychedelic Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Cannabinoid_Content as Content', out_path='Content_Plots\\Cannabinoid_UK.png', title='Cannabinoid Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Dissociative_Content as Content', out_path='Content_Plots\\Dissociative_UK.png', title='Dissociative Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Stimulant_Content as Content', out_path='Content_Plots\\Dissociative_UK.png', title='Stimulant Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Depressant_Content as Content', out_path='Content_Plots\\Dissociative_UK.png', title='Depressant Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)
plot_content(db_path, 'Other_Content as Content', out_path='Content_Plots\\Other_UK.png', title='Other Content (United Kingdom)', llcrnrlat=49.5, llcrnrlon=-12, urcrnrlat=59, urcrnrlon=4, box_size=0.5)

