"""
The main program for execution.

Project: EcstasyData
Path: root/main.py
"""
import get_data

import classifications as clss

import sqlite3

import os

db_name = 'EcstasyData.sqlite'

# get_data.create_database('https://www.ecstasydata.org/search.php?source=1&Max=5000&style=data_only', db_name)

############# PLOT CONTENT MAP #################
conn = sqlite3.connect(path)
c = conn.cursor()

c.execute("""CREATE VIEW Content_Map AS
	SELECT
		Date_Normalized,
		Latitude,
		Longitude,
		MDMA_Content,
		Enactogen_Content,
		Psychedelic_Content,
		Cannabinoid_Content,
		Dissociative_Content,
		Stimulant_Content,
		Depressant_Content,
		Other_Content
	FROM
		SOM_Classification, Location
	WHERE
		SOM_Classification.X = Location.X
		AND SOM_Classification.Y = Location.Y
		AND SOM_Classification.Z = Location.Z
	""")

conn.close()