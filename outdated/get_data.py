"""
This program fetches data from ecstacy data.org

Project: EcstasyData
Path: root/get_data.py
"""

import classifications as clss
import visualizer as vis

import urllib.request
import sqlite3

import warnings

import calendar
import datetime

import geopy
from geopy.extra.rate_limiter import RateLimiter

import time
import re

import numpy as np

import progressbar

import shutil

class SubstanceError(Exception):
    """
    Raised when there is an issue parsing substances
    """
    pass

class ClassificationError(Exception):
    """
    Raised for errors classifying substances.
    """
    pass

class TestingError(Exception):
    """
    Raised when the sample wasn't tested
    """
    pass

class LocationError(Exception):
    """
    Raised when the location isn't specified well.
    """
    pass

class UnidentifiedError(Exception):
    """
    Raised when there is an issue parsing substances
    """
    pass

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'
warnings.formatwarning = custom_formatwarning

# def build_processed_data(url):
# 	"""
# 	Returns:
# 		0: Processed data list
# 		1: List of all unknown substances
# 	"""
# 	html = _load_data(url)
# 	data_string = _remove_junk(html)
# 	processed_data = _read_to_list(data_string)
# 	return processed_data

# def create_database(url, path):
# 	processed_data  = build_processed_data(url)
# 	_list_to_database(path, processed_data)


# def _load_data(url):
# 	uf = urllib.request.urlopen(url)
# 	html_bytes = uf.read()
# 	html = html_bytes.decode('utf8')
# 	uf.close()
# 	return html

# def _remove_junk(html):
# 	"""
# 	Removes everything from the html except the data. Gets rid of blank lines and legend.

# 	Data is assumed to be between "BEGIN_DATA_BLOCK" and "END_DATA_BLOCK"

# 	Args:
# 		html: The html file corresponding to the data page.

# 	Returns:
# 		0: Data in form of a string with lines formatted as
# 			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
# 	"""
# 	start_flag = 'BEGIN_DATA_BLOCK'
# 	end_flag = 'END_DATA_BLOCK'
# 	start_index = html.find(start_flag)
# 	end_index = html.find(end_flag)
# 	no_header_footer = html[start_index+len(start_flag) : end_index]
# 	data_string = no_header_footer.replace('<br>\n', '')
# 	data_string = data_string.rstrip().lstrip()
# 	return data_string

# def _read_to_list(data_string):
# 	"""
# 	Reads a data string (as output by _remove_junk) and puts it into a list.

# 	Args:
# 		data_string: Data in form of a string with lines formatted as
# 			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
# 		warn_other: Default False. If true, warns the user every time a
# 			substance is classified as other.

# 	Returns:
# 		0: List of tuples or form:
# 		[(DataDataID integer, URL text, Name text, OtherName text,
# 		SoldAsEcstasy integer, composition, Date datetime.date,
# 		loc str, Dose integer), ...]
# 	"""
# 	good_lines = 0
# 	bad_lines = 0
# 	unknown_chem_lines = 0
# 	untested_lines= 0
# 	unknown_substances = []
# 	processed_data = []
# 	for line_index, line in enumerate(data_string.splitlines()):
# 		# The first line is column titles
# 		if line_index == 0:
# 			continue
# 		clean_line = line.lstrip().rstrip()
# 		# There are many blank lines
# 		if clean_line == '':
# 			continue
# 		raw_tuple = clean_line.split('|')
# 		try:
# 			processed_tuple = _raw_tuple_to_processed_data(raw_tuple)
# 			processed_data.append(processed_tuple)
# 			good_lines += 1
# 		except ValueError as e:
# 			bad_lines += 1
# 			warnings.warn('Could not read the following line\n'+str(raw_tuple)+'\nbecause '+str(e))
# 		except AttributeError as e:
# 			bad_lines += 1
# 			warnings.warn('Attribute issue with the following line\n'+str(raw_tuple)+'\nbecause '+str(e))
# 		except TestingError as substance_str:
# 			untested_lines += 1

# 	# Output unknown substances
# 	alphabetic_substances = list(set(unknown_substances))
# 	counted_substances = [unknown_substances.count(x) for x in alphabetic_substances]
# 	out_f = open('unknown_substances.txt', 'w')
# 	for count, name in sorted(zip(counted_substances, alphabetic_substances)):
# 		out_f.write(str(count))
# 		out_f.write('    ')
# 		out_f.write(name)
# 		out_f.write('\n')
# 	out_f.close()

# 	total_lines = good_lines + bad_lines + unknown_chem_lines+untested_lines
# 	percentages = [str(round(100*x/total_lines,1)) for x in (good_lines, bad_lines, untested_lines)]
# 	output_figures = tuple([str(total_lines)] + percentages)
# 	print('Processed %s lines. %s%% kept, %s%% poorly formatted, %s%% untested.' %\
# 		output_figures)
# 	return processed_data

# def _raw_tuple_to_processed_data(raw_tuple):
# 	"""
# 	Interprets a tuple with elements
# 			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
# 	into the form
# 		(DataDataID integer, URL text, Name text, OtherName text,
# 		SoldAsEcstasy integer, composition, Date datetime.date,
# 		loc str, Dose integer)
# 	"""
# 	DataDataID = int(raw_tuple[0])

# 	URL = raw_tuple[1]

# 	Name = raw_tuple[5]

# 	OtherName = raw_tuple[6]

# 	SoldAsEcstasy_str = raw_tuple[8]
# 	if 'probably sold as ecstasy' == SoldAsEcstasy_str:
# 		SoldAsEcstasy = 1
# 	elif 'NOT SOLD AS ECSTASY' == SoldAsEcstasy_str:
# 		SoldAsEcstasy = 0
# 	else:
# 		SoldAsEcstasy = 0
# 		warnings.warn(str(DataDataID) + ' ' + SoldAsEcstasy_str + ' unknown if sold as ecstasy.')

# 	Substance_str = raw_tuple[9]
# 	composition = _parse_substance_str(Substance_str)

# 	Date_str = raw_tuple[11]
# 	Date = _parse_date(Date_str)

# 	Location_str = raw_tuple[12]

# 	Dose_str = raw_tuple[13]
# 	Dose = _parse_dose(Dose_str)

# 	return (DataDataID, URL, Name, OtherName, SoldAsEcstasy, composition, \
# 		Date, Location_str, Dose)

# def _parse_dose(dose_str):
# 	"""
# 	Parses the dose from a dose string.

# 	Args:
# 		dose_str: Dose string

# 	Returns:
# 		0: Integer
# 	"""
# 	# If the dose was not recorded:	
# 	if dose_str == '' or '-':
# 		return 0
# 	# If we have a dose string.
# 	correctly_read = True
# 	digits = 0
# 	while correctly_read:
# 		digits += 1
# 		try:
# 			int(dose_str[:digits])
# 		except ValueError:
# 			correctly_read = False
# 	return int(dose_str[:digits-1])

def get_coordinates(location_str, no_query=False):
	"""
	Finds the x, y, z coordinates of the unit vector to a city on the globe.

	Args:
		location_str: A string describing the location.

	Returns:
		0: latitude
		1: longitude
		2: x
		3: y
		4: z
	"""
	if no_query:
		return 0,0,1,0,0
	# We need to wait one second so as not to have too many requests
	time.sleep(2)

	# Parse the location string into a query
	loc_query = location_str

	if 'nknown' in location_str:
		loc_query = loc_query.replace('Unknown, ','')
		loc_query = loc_query.replace(', Unknown','')
		loc_query = loc_query.replace('Unknown','')

	if 'nline' in location_str:
		loc_query = loc_query.replace('Online, ','')
		loc_query = loc_query.replace(', Online','')
		loc_query = loc_query.replace('Online','')

	loc_query = loc_query.rstrip().lstrip()

	if location_str.rstrip() == '' or loc_query.rstrip() == '':
		raise LocationError(location_str + ' is empty')

	# Create a client to query.
	with open('API\\TomTom') as f:
		tomtomAPI = f.readlines()[0].rstrip()
	# geolocator = geopy.geocoders.TomTom(tomtomAPI)
	geolocator = geopy.geocoders.Nominatim(user_agent='Ecstasy_Data_2018', timeout=4)

	# Geocoding an address
	try:
		location = geolocator.geocode(loc_query)
	except geopy.exc.GeocoderTimedOut:
		raise LocationError(location_str + ' interpreted as ' + loc_query)

	if location == None:
		raise LocationError(location_str)
	xyz = vis.lat_lng_to_x_y_z(location.latitude, location.longitude)
	x, y, z = xyz
	return location.latitude, location.longitude, x, y, z

# def _parse_date(date_str):
# 	"""
# 	Parses the mmm dd, yyyy format found in the html files into a datetime.date object.

# 	Args:
# 		date_str: String with format mmm dd, yyyy where mmm is a three letter month abbreviation.

# 	Returns:
# 		0: datetime.date object
# 	"""
# 	abbr_to_int = {v:k for k,v in enumerate(calendar.month_abbr)}

# 	date_split = date_str.split(' ')
# 	month_abbr = date_split[0]
# 	day = int(date_split[1][:-1])
# 	year = int(date_split[2])
# 	month = abbr_to_int[month_abbr]
# 	return datetime.date(year, month, day)

# def _parse_substance_str(substance_str):
# 	"""
# 	Parses a substance string.

# 	Args:
# 		substance_str: The substance string in the html formal

# 	Returns:
# 		0: List of substances and quantities.
# 			E.g. [('MDMA', 2), (Methamphetamine, 1)]
# 	"""
# 	# Deal with untested
# 	if substance_str in clss.aliases_for_nothing:
# 		raise TestingError(substance_str)
# 	# Deal with sugar pills
# 	if 'None Detected' in substance_str:
# 		return [('None detected',1.0)]
# 	# First split over colons
# 	substance_list = substance_str.replace('trace', '0').replace('---','0').split(';;')
# 	# All the elements take the form substance:parts
# 	# except sometimes they are given as substance:
# 	# with no number if there is only one chemical
# 	if len(substance_list) == 1 and substance_list[0][-1] == ':':
# 		substance_names = [substance_list[0][:-1]]
# 		substance_parts = [1.0]
# 	else:
# 		substance_names = []
# 		substance_parts = []
# 		for name_part in substance_list:
# 			name, part = name_part.split(':')
# 			substance_names.append(name)
# 			substance_parts.append(float(part))

# 	# Convert parts to percentages
# 	total_parts = sum(substance_parts)
# 	if total_parts < 0.1:
# 		# This exception triggers when only trace quantities of
# 		# chemicals are found (i.e. when we only have things with 0s or ---)
# 		return [('Trace detected',1.0)]

# 	return list(zip(substance_names, substance_parts))

# def _parse_substance_str(substance_str, warn_other = False):
# 	"""
# 	Parses a substance string.

# 	Args:
# 		substance_str: The substance string in the html formal
# 		warn_other: Default False. If true, warns the user every time a
# 			substance is classified as other.

# 	Returns:
# 		0: List of percentages [MDMA, Ecstasy-like substances,
# 			Psychedelics, Cannabinoids, Dissociatives, Stimulants,
# 			Depressants, Other]
# 	"""
# 	# Deal with untested
# 	if substance_str in clss.aliases_for_nothing:
# 		raise TestingError(substance_str)
# 	# Deal with sugar pills
# 	if 'None Detected' in substance_str:
# 		return [0,0,0,0,0,0,0,1]
# 	# First split over colons
# 	substance_list = substance_str.replace('trace', '0').replace('---','0').split(';;')
# 	# All the elements take the form substance:parts
# 	# except sometimes they are given as substance:
# 	# with no number if there is only one chemical
# 	if len(substance_list) == 1 and substance_list[0][-1] == ':':
# 		substance_names = [substance_list[0][:-1]]
# 		substance_parts = [1.0]
# 	else:
# 		substance_names = []
# 		substance_parts = []
# 		for name_part in substance_list:
# 			name, part = name_part.split(':')
# 			substance_names.append(name)
# 			substance_parts.append(float(part))

# 	# Convert parts to percentages
# 	total_parts = sum(substance_parts)
# 	try:
# 		substance_percentages = [x/total_parts for x in substance_parts]
# 	except ZeroDivisionError:
# 		# This exception triggers when only trace quantities of
# 		# chemicals are found (i.e. when we only have things with 0s or ---)
# 		return [0,0,0,0,0,0,0,1]

# 	# Convert substance names to substance types
# 	substance_types = [classify_substance(sub_name, warn_other=warn_other) \
# 		for sub_name in substance_names]

# 	# Join substance types together
# 	composition = [0]*9
# 	for substance_type, substance_percentage in \
# 		zip(substance_types, substance_percentages):
# 		composition[substance_type] += substance_percentage

# 	return composition[:-1]

# def classify_substance(substance, warn_other = False):
# 	"""
# 	Classifies a substance as MDMA, an ecstasy-like substances, a psychedelic,
# 	a cannabinoid, a dissociative, a stimulant, a depressant, or other.

# 	Args:
# 		substance: The substance name.
# 		unknown_other: Default False. If true, unknown chemicals will be classified as other.

# 	Returns:
# 		0: An integer. 0 for MDMA, 1 for ecstasy-like, 2 for psychedelic,
# 			3 for cannabinoid, 4 for dissociative, 5 for stimulant, 6 for depressant, 7 for other.
# 	"""
# 	if substance == 'MDMA':
# 		return 0
# 	elif substance in clss.ecstasy_like:
# 		return 1
# 	elif substance in clss.psychedelics:
# 		return 2
# 	elif substance in clss.cannabinoids:
# 		return 3
# 	elif substance in clss.dissociatives:
# 		return 4
# 	elif substance in clss.stimulants:
# 		return 5
# 	elif substance in clss.depressants:
# 		return 6
# 	elif substance in clss.silent_others:
# 		return 7
# 	elif warn_other:
# 		raise UnidentifiedError(substance)
# 	else:
# 		return 7

# def _substance_table(substance_list):
# 	substance_table = []
# 	substances = set(substance_list)
# 	unknown_substances = []
# 	for i, substance in enumerate(substances):
# 		try:
# 			classification = classify_substance(substance, warn_other=True)
# 		except UnidentifiedError as e:
# 			unknown_substances.append(str(e))
# 		classification_list = [0,0,0,0,0,0,0,0]
# 		classification_list[classification] = 1
# 		classification_tuple = tuple(classification_list)
# 		data_tup = (i, substance) + classification_tuple
# 		substance_table.append(data_tup)

# 	n_unkn = len(unknown_substances)
# 	unkn_set = set(unknown_substances)
# 	counted = [(unknown_substances.count(sub), sub) for sub in unkn_set]
# 	counted.sort()

# 	if n_unkn > 0:
# 		warnings.warn(str(n_unkn) + ' instances of ' + str(len(unkn_set)) + \
# 			' unknown substances saved to unknown_substances.txt')
# 		out_f = open('unknown_substances.txt', 'w')
# 		for instances, substance in counted:
# 			out_f.write(str(instances))
# 			out_f.write('     ')
# 			out_f.write(substance)
# 			out_f.write('\n')

# 	return substance_table

# def _list_to_database(path, data_list):
# 	"""
# 	Converts a list to a database with sqlite3.

# 	Args:
# 		path: Path to store the database.
# 		data_list: List of tuples or form:
# 		[(DataDataID integer, URL text, Name text, OtherName text,
# 		SoldAsEcstasy integer, composition, Date datetime.datetime,
# 		loc str, Dose integer), ...]
# 	"""
# 	conn = sqlite3.connect(path)
# 	c = conn.cursor()

# 	# TABLE Location
# 	# Location_ID | Location_String | Latitude | Longitude | X | Y | Z
# 	c.execute("""CREATE TABLE Location (
# 			Location_ID integer,
# 			Location_String text,
# 			Latitude real,
# 			Longitude real,
# 			X real,
# 			Y real,
# 			Z real
# 		);""")
# 	location_list = [foo[7] for foo in data_list]
# 	location_list = list(set(location_list))
# 	bad_locations = []
# 	print('Querying for locations')
# 	bar = progressbar.ProgressBar(max_value=len(location_list), redirect_stdout=True).start()
# 	for i, location in enumerate(location_list):
# 		try:
# 			lat, lng, x, y, z = get_coordinates(location)
# 			c.execute('INSERT INTO Location VALUES (?,?,?,?,?,?,?);', (i, location, lat, lng, x, y, z))
# 		except LocationError as loc:
# 			print('Bad location: ' + str(loc))
# 			bad_locations.append(str(loc))
# 		bar.update(i+1)
# 	print('Found', len(bad_locations), 'bad location lines')
# 	out_f = open('unknown_locations.txt', 'w')
# 	for loc in bad_locations:
# 		out_f.write(loc)
# 		out_f.write('\n')
# 	out_f.close()
# 	bar.finish()

# 	# TABLE Substances
# 	# Substance_ID | Substance_Name | MDMA | Enactogen | Psychedelic | Cannabinoid | Dissocciative |
# 	# Stimulant | Depressant | Other
# 	c.execute("""CREATE TABLE Substance (
# 		Substance_ID integer,
# 		Substance_Name text,
# 		MDMA real,
# 		Enactogen real,
# 		Psychedelic real,
# 		Cannabinoid real,
# 		Dissociative real,
# 		Stimulant real,
# 		Depressant real,
# 		Other real
# 		);""")
# 	substance_list = clss.ecstasy_like + clss.psychedelics + clss.cannabinoids + \
# 		clss.dissociatives + clss.stimulants + clss.depressants + clss.silent_others
# 	for data_tup in data_list:
# 		substance_names = list(list(zip(*data_tup[5]))[0])
# 		substance_list += substance_names
# 	substance_table = _substance_table(substance_list)
# 	for substance_data_tuple in substance_table:
# 		c.execute('INSERT INTO Substance VALUES (?,?,?,?,?,?,?,?,?,?);', substance_data_tuple)

# 	# TABLE Pill_Content
# 	# Pill_ID | SubstanceID | Substance_Parts | Substance_Percentage
# 	c.execute("""CREATE TABLE Pill_Content_Names (
# 		Pill_ID integer,
# 		Substance_Name text,
# 		Substance_Parts real,
# 		Substance_Percentage real
# 		);""")
# 	for data_tup in data_list:
# 		pill_id = data_tup[0]
# 		composition = data_tup[5]
# 		total_parts = sum(x[1] for x in composition)
# 		for substance, part in composition:
# 			percentage = part/total_parts
# 			c.execute('INSERT INTO Pill_Content_Names VALUES (?,?,?,?);', (pill_id, substance, part, percentage))

# 	# Now we need to change all the names into ids
# 	c.execute("""CREATE TABLE Pill_Content AS
# 		SELECT
# 			Pill_Content_Names.Pill_ID,
# 			Substance.Substance_ID,
# 			Pill_Content_Names.Substance_Parts,
# 			Pill_Content_Names.Substance_Percentage
# 		FROM
# 			Pill_Content_Names, Substance
# 		WHERE
# 			Pill_Content_Names.Substance_Name = Substance.Substance_Name;""")

# 	c.execute('DROP TABLE Pill_Content_Names;')

# 	# TABLE Pill_Misc
# 	# Pill_ID | Location_ID | Date_Normalized | Sold_As_Ecstasy | Date | URL | Name | Other_Name | Dose
# 	c.execute("""CREATE TABLE Pill_Misc_Name (
# 		Pill_ID integer,
# 		Location_String text,
# 		Date_Normalized real,
# 		Sold_As_Ecstasy integer,
# 		Date text,
# 		URL text,
# 		Name text,
# 		Other_Name text,
# 		Dose integer
# 		);""")
# 	# (DataDataID integer, URL text, Name text, OtherName text,
# 	# 	SoldAsEcstasy integer, composition, Date datetime.datetime,
# 	# 	loc str, Dose integer)
# 	dates = [x[6].toordinal() for x in data_list]
# 	start_date = min(dates)
# 	end_date = max(dates)
# 	def __normalize_date(date): return (date.toordinal()-start_date)/(end_date-start_date)
# 	for data_tup in data_list:
# 		pill_id = data_tup[0]
# 		location_name = data_tup[7]
# 		sold_as_ecstasy = data_tup[4]
# 		date = data_tup[6]
# 		date_normalized = __normalize_date(date)
# 		url = data_tup[1]
# 		name = data_tup[2]
# 		other_name = data_tup[3]
# 		dose = data_tup[8]
# 		table_tuple = (pill_id, location_name, date_normalized, sold_as_ecstasy, \
# 			date, url, name, other_name, dose)
# 		c.execute('INSERT INTO Pill_Misc_Name VALUES (?,?,?,?,?,?,?,?,?);', table_tuple)

# 	c.execute(""" CREATE TABLE Pill_Misc AS
# 		SELECT
# 			Pill_Misc_Name.Pill_ID,
# 			Location.Location_ID,
# 			Pill_Misc_Name.Date_Normalized,
# 			Pill_Misc_Name.Sold_As_Ecstasy,
# 			Pill_Misc_Name.Date,
# 			Pill_Misc_Name.URL,
# 			Pill_Misc_Name.Name,
# 			Pill_Misc_Name.Other_Name,
# 			Pill_Misc_Name.Dose
# 		FROM
# 			Pill_Misc_Name, Location
# 		WHERE
# 			Pill_Misc_Name.Location_String = Location.Location_String
# 		;""")

# 	c.execute('DROP TABLE Pill_Misc_Name')

# 	# VIEW SOM_Data
# 	# Pill_ID | Date | X | Y | Z | MDMA_Content | Enactogen_Content | Psychedelic_Content
# 	# | Cannabinoid_Content | Dissociative_Content | Stimulant_Content | Depressant_Content
# 	# | Other
# 	c.execute("""CREATE VIEW SOM_Classification AS
# 		SELECT
# 		    Pill_Misc.Pill_ID,
# 		    Pill_Misc.Date_Normalized,
# 		    Location.X,
# 		    Location.Y,
# 		    Location.Z,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.MDMA ) AS MDMA_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Enactogen) AS Enactogen_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Psychedelic) AS Psychedelic_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Cannabinoid ) AS Cannabinoid_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Dissociative) AS Dissociative_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Stimulant ) AS Stimulant_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Depressant) AS Depressant_Content,
# 		    SUM(Pill_Content.Substance_Percentage * Substance.Other) AS Other_Content
# 		FROM
# 		    Pill_Misc, Pill_Content, Substance, Location
# 		WHERE Pill_Misc.Pill_ID = Pill_Content.Pill_ID
# 		AND Pill_Misc.Location_ID = Location.Location_ID
# 		AND Pill_Content.Substance_ID = Substance.Substance_ID
# 		AND Pill_Misc.SoldAsEcstasy = 1
# 		GROUP BY Pill_Misc.Pill_ID
# 	;""")

# 	c.execute("""CREATE VIEW Content_Map AS
# 		SELECT
# 			Date_Normalized,
# 			Latitude,
# 			Longitude,
# 			MDMA_Content,
# 			Enactogen_Content,
# 			Psychedelic_Content,
# 			Cannabinoid_Content,
# 			Dissociative_Content,
# 			Stimulant_Content,
# 			Depressant_Content,
# 			Other_Content
# 		FROM
# 			SOM_Classification, Location
# 		WHERE
# 			SOM_Classification.X = Location.X
# 			AND SOM_Classification.Y = Location.Y
# 			AND SOM_Classification.Z = Location.Z;""")

# 	conn.commit()
# 	conn.close()

class EDataDB:
	# Class for interacting with ecstasy data database

	# TABLE Location
	# Location_ID | Location_Name | Latitude | Longitude | X | Y | Z

	# TABLE Substance
	# Substance_ID | Substance_Name | MDMA | Enactogen | Psychedelic | Cannabinoid | Dissocciative |
	# Stimulant | Depressant | Other

	# TABLE Pill_Content
	# Pill_ID | SubstanceID | Substance_Parts | Substance_Percentage

	# TABLE Pill_Misc
	# Pill_ID | Location_ID | Date_Normalized | Sold_As_Ecstasy | Date | URL | Name | Other_Name | Dose

	# VIEW SOM_Classification
	# Pill_ID | MDMA_Content | Enactogen_Content | Psychedelic_Content
	# | Cannabinoid_Content | Dissociative_Content | Stimulant_Content | Depressant_Content | Other_Content

	# VIEW SOM_Data
	# Date_Normalized | X | Y | Z | MDMA_Content | Enactogen_Content | Psychedelic_Content
	# | Cannabinoid_Content | Dissociative_Content | Stimulant_Content | Depressant_Content | Other_Content

	# VIEW Conten_Map
	# Date_Normalized | Latitude | Longitude | MDMA_Content | Enactogen_Content | Psychedelic_Content
	# | Cannabinoid_Content | Dissociative_Content | Stimulant_Content | Depressant_Content | Other_Content
	def __init__(self, path):
		"""
		Loads an existing database or creates one in path
		"""
		self.path = path
		self.unknown_substance_count = dict()
		self.bad_locations = set()

	def new_copy(self, copy_path):
		# See if the database is already open
		try:
			self.conn.close()
		except AttributeError:
			pass
		# Copy the database to a new location
		shutil.copy(self.path, copy_path)
		# Update self.path
		self.path = copy_path

	def open(self):
		self.conn = sqlite3.connect(self.path)
		self.c = self.conn.cursor()

	def dump_unknown_substances(self, path):
		substances = self.unknown_substance_count.keys()
		counts = [self.unknown_substance_count[substance] for substance in substances]
		zipper = list(zip(counts, substances))
		zipper.sort()
		out_f = open(path, 'w')
		for count, substance in zipper:
			out_f.write(str(count))
			out_f.write('    ')
			out_f.write(substance)
			out_f.write('\n')
		out_f.close()

	def dump_bad_locations(self, path):
		out_f = open(path, 'w')
		for locale in self.bad_locations:
			out_f.write(locale)
			out_f.write('\n')
		out_f.close()

	def initialize(self):
		"""
		Creates any missing database table.
		"""

		# Location
		self.c.execute("""
			CREATE TABLE IF NOT EXISTS Location (
				Location_ID INTEGER PRIMARY KEY,
				Location_Name TEXT UNIQUE,
				Latitude real,
				Longitude real,
				X real,
				Y real,
				Z real
			);
		""")

		# Substance
		self.c.execute("""
			CREATE TABLE IF NOT EXISTS Substance (
				Substance_ID INTEGER PRIMARY KEY,
				Substance_Name text UNIQUE,
				MDMA real,
				Enactogen real,
				Psychedelic real,
				Cannabinoid real,
				Dissociative real,
				Stimulant real,
				Depressant real,
				Other real
			);
		""")

		# Pill_Content
		self.c.execute("""	
			CREATE TABLE IF NOT EXISTS Pill_Content (
				Pill_ID INTEGER PRIMARY KEY,
				Substance_ID integer,
				Substance_Parts real,
				Substance_Percentage real
			);
		""")

		# Pill_Misc
		self.c.execute("""
			CREATE TABLE IF NOT EXISTS Pill_Misc (
				Pill_ID INTEGER PRIMARY KEY,
				Location_ID integer,
				Sold_As_Ecstasy integer,
				Date text,
				URL text,
				Name text,
				Other_Name text,
				Dose integer
			);
		""")

		# SOM_Classification
		self.c.execute("""
			CREATE VIEW IF NOT EXISTS SOM_Classification AS
				SELECT
				    Pill_Misc.Pill_ID,
				    SUM(Pill_Content.Substance_Percentage * Substance.MDMA ) AS MDMA_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Enactogen) AS Enactogen_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Psychedelic) AS Psychedelic_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Cannabinoid ) AS Cannabinoid_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Dissociative) AS Dissociative_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Stimulant ) AS Stimulant_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Depressant) AS Depressant_Content,
				    SUM(Pill_Content.Substance_Percentage * Substance.Other) AS Other_Content
				FROM
				    Pill_Misc, Pill_Content, Substance, Location
				WHERE Pill_Misc.Pill_ID = Pill_Content.Pill_ID
					AND Pill_Content.Substance_ID = Substance.Substance_ID
					AND Pill_Misc.SoldAsEcstasy = 1
					AND Location.X IS NOT NULL
					AND Location.Y IS NOT NULL
					AND Location.Z IS NOT NULL
				GROUP BY Pill_Misc.Pill_ID;
		""")

		# SOM_Data
		self.c.execute("""
			CREATE VIEW IF NOT EXISTS SOM_Data AS
				SELECT
					Date,
				    X,
				    Y,
				    Z,
				    MDMA_Content,
				    Enactogen_Content,
				    Psychedelic_Content,
				    Cannabinoid_Content,
				    Dissociative_Content,
				    Stimulant_Content,
				    Depressant_Content,
				    Other_Content
				FROM
				    SOM_Classification, Pill_Misc, Location
				WHERE
					Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
					AND Location.Location_ID = Pill_Misc.Location_ID;
		""")
		
		# Content_Map
		self.c.execute("""
			CREATE VIEW IF NOT EXISTS Content_Map AS
				SELECT
					Date,
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
				    SOM_Classification, Pill_Misc, Location
				WHERE
					Pill_Misc.Pill_ID = SOM_Classification.Pill_ID
					AND Location.Location_ID = Pill_Misc.Location_ID
			""")

	def add_location(self, location_string):
		"""
		Adds a location to the location table. Querying the geocoder if necessary.

		Returns:
			0 : Integer
				The ID of the location.
		"""
		# First find out if location_name is in the Location table.
		self.c.execute('SELECT Location_ID, Location_Name FROM Location')
		location_data = self.c.fetchall()
		all_IDs, all_names = zip(*location_data)
		if location_string not in all_names:
			# We don't have this location in the database. Try to geocode it
			try:
				latitude, longitude, x, y, z = get_coordinates(location_string)
				self.c.execute("""
					INSERT INTO Location
					VALUES (null, ?, ?, ?, ?, ?, ?)
				""", (location_string, latitude, longitude, x, y, z))
			except LocationError as e:
				warnings.warn('Could not find the coordinates of', e)
				self.bad_locations.add(location_string)
				self.c.execute("""
					INSERT INTO Location
					VALUES (null, ?, null, null, null, null, null),
				""", (location_string,))
			# Get the key. It's the old maximum key plus one.
			return max(all_IDs) + 1
		else:
			location_ID = all_IDs[all_names.index(location_string)]
			return location_ID

	def add_substance(self, substance_string, classifier = None):
		"""
		Adds a substance to the Substance table.

		Args:
			substance_string : string
				This is the name of the substance
			classification : optional. Default None.
				The classification to use if the substance is not already in
				the database. If none is given, self.classifier is used.

		Raises:
			ClassificationError: If we can't classify an unknown substance.

		Returns:
			0 : Integer
				The substance ID for this substance.
		"""
		if classifier == None:
			classifier = self.classifier
		# First get the data from the table to see if we have the substance.
		self.c.execute('SELECT Substance_ID, Substance_Name FROM Substance')
		substance_data = self.c.fetchall()
		if len(substance_data) == 0:
			all_IDs = [0]
			all_names = []
		else:
			all_IDs, all_names = zip(*substance_data)
		if substance_string not in all_names:
			# We don't have this substance
			try:
				classification = classifier(substance_string)
			except ClassificationError:
				raise ClassificationError(substance_string)
			insertion_tuple = (substance_string,) + classification
			self.c.execute("""
				INSERT INTO Substance
				VALUES (null, ?, ?, ?, ?, ?, ?, ?, ?, ?);
			""", insertion_tuple)
			# Get the key. It's the old maximum + 1
			return max(all_IDs) + 1
		else:
			substance_ID = all_IDs[all_names.index(substance_string)]
			return substance_ID

	def load_html_line(self, line):
		"""
		Parses data in the form,
		DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
		
		and adds it to the database.
		"""
		raw_tuple = line.split('|')

		pill_ID = int(raw_tuple[0])

		sold_as_ecstasy_str = raw_tuple[8]
		if 'probably sold as ecstasy' == sold_as_ecstasy_str:
			sold_as_ecstasy = 1
		elif 'NOT SOLD AS ECSTASY' == sold_as_ecstasy_str:
			sold_as_ecstasy = 0
		else:
			sold_as_ecstasy = 2

		date_str = raw_tuple[11]
		date = self._parse_date(date_str)

		URL = raw_tuple[1]

		name = raw_tuple[5]

		other_name = raw_tuple[6]

		dose_str = raw_tuple[13]
		dose = self._parse_dose(dose_str)

		substance_str = raw_tuple[9]
		composition = self._parse_substance_str(substance_str)

		location_str = raw_tuple[12]

		# Begin with TABLE Pill_Misc. First we need to add the location
		location_ID = self.add_location(location_str)

		# Now add a row to Pill_Misc
		dose_format = '?'
		if dose == 0: dose_format = 'null'
		pill_misc_format = '(?, ?, %s, ?, ?, ?, ?, %s)' % (['?', '?', 'null'][sold_as_ecstasy], dose_format)
		pill_misc_data = [pill_ID, location_ID]
		if sold_as_ecstasy != 2:
			pill_misc_data.append(sold_as_ecstasy)
		pill_misc_data += [date, URL, name, other_name]
		if dose != 0:
			pill_misc_data.append(dose)
		pill_misc_data = tuple(pill_misc_data)
		self.c.execute('INSERT INTO Pill_Misc VALUES '+pill_misc_format+';', pill_misc_data)

		# Now we want to add to Pill_Content.
		for substance, parts, percentage in composition:
			# Add the substance to the Substance table
			substance_ID = self.add_substance(substance)
			self.c.execute('INSERT INTO Pill_Content VALUES (?, ?, ?, ?);', (pill_ID, substance_ID, parts, percentage))

	def classifier(self, substance_str, warn_other=False):
		"""
		Classifies a substance as MDMA, an ecstasy-like substances, a psychedelic,
		a cannabinoid, a dissociative, a stimulant, a depressant, or other.

		Args:
			substance: The substance name.
			unknown_other: Default False. If true, unknown chemicals will be classified as other.

		Returns:
			0: A one hot tuple. Hot index 0 for MDMA, 1 for ecstasy-like, 2 for psychedelic,
				3 for cannabinoid, 4 for dissociative, 5 for stimulant, 6 for depressant, 7 for other.
		"""
		if substance_str in clss.MDMA:
			classification_index = 0
		elif substance_str in clss.ecstasy_like:
			classification_index = 1
		elif substance_str in clss.psychedelics:
			classification_index = 2
		elif substance_str in clss.cannabinoids:
			classification_index = 3
		elif substance_str in clss.dissociatives:
			classification_index = 4
		elif substance_str in clss.stimulants:
			classification_index = 5
		elif substance_str in clss.depressants:
			classification_index = 6
		elif substance_str in clss.silent_others:
			classification_index = 7
		elif warn_other:
			raise ClassificationError(substance)
		else:
			classification_index = 7
			if substance_str not in self.unknown_substance_count.keys():
				self.unknown_substance_count[substance_str] = 0
			self.unknown_substance_count[substance_str] += 1
		one_hot = [0]*8
		one_hot[classification_index] = 1
		return tuple(one_hot)

	def _parse_date(self, date_str):
		"""
		Parses the mmm dd, yyyy format found in the html files into a datetime.date object.

		Args:
			date_str: String with format mmm dd, yyyy where mmm is a three letter month abbreviation.

		Returns:
			0: datetime.date object
		"""
		abbr_to_int = {v:k for k,v in enumerate(calendar.month_abbr)}

		date_split = date_str.split(' ')
		month_abbr = date_split[0]
		day = int(date_split[1][:-1])
		year = int(date_split[2])
		month = abbr_to_int[month_abbr]
		return datetime.date(year, month, day)

	def _parse_dose(self, dose_str):
		"""
		Parses the dose from a dose string.

		Args:
			dose_str: Dose string

		Returns:
			0: Integer
		"""
		# If the dose was not recorded:	
		if dose_str == '' or '-':
			return 0
		
		# Get the numeric part
		try:
			mg_dose = int(dose_str[:dose_str.index(' ')])
			return mg_dose
		except:
			warnings.warn('Could not read dosage ' + dose_str)
			return 0

	def _parse_substance_str(self, substance_str):
		"""
		Parses a substance string.

		Args:
			substance_str: The substance string in the html formal

		Returns:
			0: List of substances and quantities.
				E.g. [('MDMA', 2, 0.16), (Methamphetamine, 1, 0.08)]
		"""
		# Deal with untested
		aliases_for_nothing = ['Not tested', 'Not Tested', '', 'Not tested:', 'Not Tested:', 'Not tested:---', 'Not Tested:---', \
			'Not tested:1', 'Not Tested:1', 'On Hold - See Note:0', 'On Hold - See Note:', 'On Hold - See Note:1']
		if substance_str in aliases_for_nothing:
			raise TestingError(substance_str)
		# Deal with sugar pills
		if 'None Detected' in substance_str:
			return [('None detected', 1.0, 1.0)]
		# First split over double semicolons
		substance_list = substance_str.replace('trace', '0').replace('---','0').split(';;')
		# All the elements take the form substance:parts
		# except sometimes they are given as substance:
		# with no number if there is only one chemical
		if len(substance_list) == 1 and substance_list[0][-1] == ':':
			substance_names = [substance_list[0][:-1]]
			substance_parts = [1.0]
		else:
			substance_names = []
			substance_parts = []
			for name_part in substance_list:
				name, part = name_part.split(':')
				substance_names.append(name)
				substance_parts.append(float(part))

		# Convert parts to percentages
		total_parts = sum(substance_parts)
		if total_parts < 0.1:
			# This exception triggers when only trace quantities of
			# chemicals are found (i.e. when we only have things with 0s or ---)
			return [('Trace detected', 1.0, 1.0)]
		substance_percentages = [x/total_parts for x in substance_parts]

		return list(zip(substance_names, substance_parts, substance_percentages))

	def commit(self):

		self.conn.commit()

	def close(self):

		self.conn.close()

	def _read_url(url):
		"""
		Reads data from an EcstasyData dataonly format page. Data is assumed to
		be between "BEGIN_DATA_BLOCK" and "END_DATA_BLOCK"

		Args:
			url: The url to the hrml file corresponding to the data page.

		Returns:
			0: List of html lines that can be loaded with load_html_line
		"""
		uf = urllib.request.urlopen(url)
		html_bytes = uf.read()
		html = html_bytes.decode('utf8')
		uf.close()

		start_flag = 'BEGIN_DATA_BLOCK'
		end_flag = 'END_DATA_BLOCK'
		start_index = html.find(start_flag)
		end_index = html.find(end_flag)
		no_header_footer = html[start_index+len(start_flag) : end_index]
		data_string = no_header_footer.replace('<br>\n', '')
		data_string = data_string.rstrip().lstrip()
		return data_string.splitlines()

	def load_from_data_only_url(url):
		"""
		Loads data from an EcstasyData webpage that needs no url formatting.
		"""
		data_lines = self._read_url(url)

		# Add each data line to the database
		bar = progressbar.ProgressBar(max_value=len(data_lines), redirect_stdout=True).start()
		for line_no, line in enumerate(data_lines):
			bar.update(line_no)
			self.load_html_line(line)
		bar.finish()

	def load_from_url(url):
		"""
		Loads data from an EcstasyData webpage. E.g. Loads from https://www.ecstasydata.org/results.php.
		"""
		# First we need to add the right things to the url, i.e. &Max=5000&style=data_only
		# but max needs to be high enough to hold all the data.
		# FINDING OUT HOW MANY DATA POINTS THERE ARE:
		# First load the url
		uf = urllib.request.urlopen(url)
		html_bytes = uf.read()
		html = html_bytes.decode('utf8')
		uf.close()
		# Look for <li class="Results">_____ entries total</li>. The underscores are the location of the
		li_tag = re.compile('<li class="Results">[0-9]+ entries total</li>')
		matching_tag = li_tag.search(html).group()
		# total number of data points.
		integer_re = re.compile('[0-9]+')
		n_data = integer_re.search(html).group()
		# Add the proper flags to the url and load
		self.load_from_data_only_url(url+'?&Max=%s&style=data_only' % (n_data,))