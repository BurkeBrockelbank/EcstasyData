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

from geopy import Nominatim

import numpy as np

class SubstanceError(Exception):
    """
    Raised when there is an issue parsing substances
    """
    pass

class TestingError(Exception):
    """
    Raised when the sample wasn't tested
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

def build_processed_data(url):
	"""
	Returns:
		0: Processed data list
		1: List of all unknown substances
	"""
	html = _load_data(url)
	data_string = _remove_junk(html)
	processed_data = _read_to_list(data_string)
	return processed_data

def create_database(url, path):
	processed_data  = build_processed_data(url)
	_list_to_database(path, processed_data)


def _load_data(url):
	uf = urllib.request.urlopen(url)
	html_bytes = uf.read()
	html = html_bytes.decode('utf8')
	uf.close()
	return html

def _remove_junk(html):
	"""
	Removes everything from the html except the data. Gets rid of blank lines and legend.

	Data is assumed to be between "BEGIN_DATA_BLOCK" and "END_DATA_BLOCK"

	Args:
		html: The html file corresponding to the data page.

	Returns:
		0: Data in form of a string with lines formatted as
			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
	"""
	start_flag = 'BEGIN_DATA_BLOCK'
	end_flag = 'END_DATA_BLOCK'
	start_index = html.find(start_flag)
	end_index = html.find(end_flag)
	no_header_footer = html[start_index+len(start_flag) : end_index]
	data_string = no_header_footer.replace('<br>\n', '')
	data_string = data_string.rstrip().lstrip()
	return data_string

def _read_to_list(data_string):
	"""
	Reads a data string (as output by _remove_junk) and puts it into a list.

	Args:
		data_string: Data in form of a string with lines formatted as
			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
		warn_other: Default False. If true, warns the user every time a
			substance is classified as other.

	Returns:
		0: List of tuples or form:
		[(DataDataID integer, URL text, Name text, OtherName text,
		SoldAsEcstasy integer, composition, Date datetime.date,
		loc str, Dose integer), ...]
	"""
	good_lines = 0
	bad_lines = 0
	unknown_chem_lines = 0
	untested_lines= 0
	unknown_substances = []
	processed_data = []
	for line_index, line in enumerate(data_string.splitlines()):
		# The first line is column titles
		if line_index == 0:
			continue
		clean_line = line.lstrip().rstrip()
		# There are many blank lines
		if clean_line == '':
			continue
		raw_tuple = clean_line.split('|')
		try:
			processed_tuple = _raw_tuple_to_processed_data(raw_tuple)
			processed_data.append(processed_tuple)
			good_lines += 1
		except ValueError as e:
			bad_lines += 1
			warnings.warn('Could not read the following line\n'+str(raw_tuple)+'\nbecause '+str(e))
		except AttributeError as e:
			bad_lines += 1
			warnings.warn('Attribute issue with the following line\n'+str(raw_tuple)+'\nbecause '+str(e))
		except TestingError as substance_str:
			untested_lines += 1

	# Output unknown substances
	alphabetic_substances = list(set(unknown_substances))
	counted_substances = [unknown_substances.count(x) for x in alphabetic_substances]
	out_f = open('unknown_substances.txt', 'w')
	for count, name in sorted(zip(counted_substances, alphabetic_substances)):
		out_f.write(str(count))
		out_f.write('    ')
		out_f.write(name)
		out_f.write('\n')
	out_f.close()

	total_lines = good_lines + bad_lines + unknown_chem_lines+untested_lines
	percentages = [str(round(100*x/total_lines,1)) for x in (good_lines, bad_lines, untested_lines)]
	output_figures = tuple([str(total_lines)] + percentages)
	print('Processed %s lines. %s%% kept, %s%% poorly formatted, %s%% untested.' %\
		output_figures)
	return processed_data

def _raw_tuple_to_processed_data(raw_tuple):
	"""
	Interprets a tuple with elements
			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
	into the form
		(DataDataID integer, URL text, Name text, OtherName text,
		SoldAsEcstasy integer, composition, Date datetime.date,
		loc str, Dose integer)
	"""
	DataDataID = int(raw_tuple[0])

	URL = raw_tuple[1]

	Name = raw_tuple[5]

	OtherName = raw_tuple[6]

	SoldAsEcstasy_str = raw_tuple[8]
	if 'probably sold as ecstasy' == SoldAsEcstasy_str:
		SoldAsEcstasy = 1
	elif 'NOT SOLD AS ECSTASY' == SoldAsEcstasy_str:
		SoldAsEcstasy = 0
	else:
		SoldAsEcstasy = 0
		warnings.warn(str(DataDataID) + ' ' + SoldAsEcstasy_str + ' unknown if sold as ecstasy.')

	Substance_str = raw_tuple[9]
	composition = _parse_substance_str(Substance_str)

	Date_str = raw_tuple[11]
	Date = _parse_date(Date_str)

	Location_str = raw_tuple[12]

	Dose_str = raw_tuple[13]
	Dose = _parse_dose(Dose_str)

	return (DataDataID, URL, Name, OtherName, SoldAsEcstasy, composition, \
		Date, Location_str, Dose)

def _parse_dose(dose_str):
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
	# If we have a dose string.
	correctly_read = True
	digits = 0
	while correctly_read:
		digits += 1
		try:
			int(dose_str[:digits])
		except ValueError:
			correctly_read = False
	return int(dose_str[:digits-1])

def _get_coordinates(location_str):
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
	# # Create a client to query.
	# geolocator = Nominatim(user_agent='Ecstasy_Data_2018')
	# # Geocoding an address
	# location = geolocator.geocode(location_str)
	# xyz = vis.lat_lng_to_x_y_z(location.latitude, location.longitude)
	# x, y, z = xyz
	# return location.latitude, location.longitude, x, y, z
	return 0, 0, 1, 0, 0

def _parse_date(date_str):
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

def _parse_substance_str(substance_str):
	"""
	Parses a substance string.

	Args:
		substance_str: The substance string in the html formal

	Returns:
		0: List of substances and quantities.
			E.g. [('MDMA', 2), (Methamphetamine, 1)]
	"""
	# Deal with untested
	if substance_str in clss.aliases_for_nothing:
		raise TestingError(substance_str)
	# Deal with sugar pills
	if 'None Detected' in substance_str:
		return []
	# First split over colons
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
		return []

	return list(zip(substance_names, substance_parts))

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

def classify_substance(substance, warn_other = False):
	"""
	Classifies a substance as MDMA, an ecstasy-like substances, a psychedelic,
	a cannabinoid, a dissociative, a stimulant, a depressant, or other.

	Args:
		substance: The substance name.
		unknown_other: Default False. If true, unknown chemicals will be classified as other.

	Returns:
		0: An integer. 0 for MDMA, 1 for ecstasy-like, 2 for psychedelic,
			3 for cannabinoid, 4 for dissociative, 5 for stimulant, 6 for depressant, 7 for other.
	"""
	if substance == 'MDMA':
		return 0
	elif substance in clss.ecstasy_like:
		return 1
	elif substance in clss.psychedelics:
		return 2
	elif substance in clss.cannabinoids:
		return 3
	elif substance in clss.dissociatives:
		return 4
	elif substance in clss.stimulants:
		return 5
	elif substance in clss.depressants:
		return 6
	elif substance in clss.silent_others:
		return 7
	elif warn_other:
		raise UnidentifiedError(substance)
	else:
		return 7

def _substance_table(substance_list):
	substance_table = []
	substances = set(substance_list)
	for i, substance in enumerate(substance_list):
		classification = classify_substance(substance)
		classification_list = [0,0,0,0,0,0,0,0]
		classification_list[classification] = 1
		classification_tuple = tuple(classification_list)
		data_tup = (i, substance) + classification_tuple
		substance_table.append(data_tup)
	return substance_table

def _list_to_database(path, data_list):
	"""
	Converts a list to a database with sqlite3.

	Args:
		path: Path to store the database.
		data_list: List of tuples or form:
		[(DataDataID integer, URL text, Name text, OtherName text,
		SoldAsEcstasy integer, composition, Date datetime.datetime,
		loc str, Dose integer), ...]
	"""
	conn = sqlite3.connect(path)
	c = conn.cursor()

	# TABLE Location
	# Location_ID | Location_String | Latitude | Longitude | X | Y | Z
	c.execute("""CREATE TABLE Location (
			Location_ID integer,
			Location_String text,
			Latitude real,
			Longitude real,
			X real,
			Y real,
			Z real
		);""")
	location_list = [foo[7] for foo in data_list]
	location_list = list(set(location_list))
	for i, location in enumerate(location_list):
		lat, lng, x, y, z = _get_coordinates(location)
		c.execute('INSERT INTO Location VALUES (?,?,?,?,?,?,?);', (i, location, lat, lng, x, y, z))

	# TABLE Substances
	# Substance_ID | Substance_Name | MDMA | Enactogen | Psychedelic | Cannabinoid | Dissocciative |
	# Stimulant | Depressant | Other
	c.execute("""CREATE TABLE Substance (
		Substance_ID integer,
		Substance_Name text,
		MDMA real,
		Enactogen real,
		Psychedelic real,
		Cannabinoid real,
		Dissociative real,
		Stimulant real,
		Depressant real,
		Other real
		);""")
	substance_list = clss.ecstasy_like + clss.psychedelics + clss.cannabinoids + \
		clss.dissociatives + clss.stimulants + clss.depressants + clss.silent_others
	for data_tup in data_list:
		substance_names = list(list(zip(*data_tup[5]))[0])
		substance_list += substance_names
	substance_table = _substance_table(substance_list)
	for substance_data_tuple in substance_table:
		c.execute('INSERT INTO Substance VALUES (?,?,?,?,?,?,?,?,?,?);', substance_data_tuple)

	# TABLE Pill_Content
	# Pill_ID | SubstanceID | Substance_Parts | Substance_Percentage
	c.execute("""CREATE TABLE Pill_Content_Names (
		Pill_ID integer,
		Substance_Name text
		Substance_Parts real,
		Substance_Percentage real
		);""")
	for data_tup in data_list:
		pill_id = dat_tup[0]
		composition = data_tup[5]
		total_parts = sum(x[1] for x in composition)
		for substance, part in composition:
			percentage = part/total_parts
			c.execute('INSERT INTO Pill_Content VALUES (?,?,?,?);', (pill_id, substance, part, percentage))
	# Now we need to change all the names into ids
	c.execute("""CREATE TABLE Pill_Content AS
		SELECT
			Pill_Content_Names.Pill_ID,
			Substance.Substance_ID
			Pill_Content_Names.Substance_Parts,
			Pill_Content_Names.Substance_Percentage
		FROM
			Pill_Content_Names, Substance
		WHERE
			Pill_Content_Names.Substance_Name = Substance.Substance_Name
		;""")
	c.execute('DROP TABLE Pill_Content_Names;')

	# TABLE Pill_Misc
	# Pill_ID | Location_ID | Date_Normalized | Sold_As_Ecstasy | Date | URL | Name | Other_Name | Dose
	c.execute("""CREATE TABLE Pill_Misc_Name (
		Pill_ID integer,
		Location_Name text,
		Date_Normalized real,
		Sold_As_Ecstasy integer,
		Date text,
		URL text,
		Name text,
		Other_Name text,
		Dose integer
		);""")
	# (DataDataID integer, URL text, Name text, OtherName text,
	# 	SoldAsEcstasy integer, composition, Date datetime.datetime,
	# 	loc str, Dose integer)
	dates = [x[6] for x in data_list]
	start_date = min(dates).toordinal()
	end_date = max(dates).toordinal()
	def __normalize_date(date): return (date.toordinal()-x)/(y-x)
	for data_tup in data_list:
		pill_id = dat_tup[0]
		location_name = dat_tup[7]
		sold_as_ecstasy = dat_tup[4]
		date = dat_tup[6]
		date_normalized = __normalize_date(date)
		url = dat_tup[1]
		name = dat_tup[2]
		other_name = dat_tup[3]
		dose = dat_tup[8]
		table_tuple = (pill_id, location_name, date_normalized, sold_as_ecstasy, \
			date, url, name, other_name, dose)
		c.execute('INSERT INTO Pill_Misc_Name VALUES (?,?,?,?,?,?,?,?,?);', table_tuple)

	c.execute(""" CREATE TABLE Pill_Misc AS
		SELECT
			Pill_Misc_Name.Pill_ID,
			Location.Location_ID,
			Pill_Misc_Name.Date_Normalized,
			Pill_Misc_Name.Sold_As_Ecstasy,
			Pill_Misc_Name.Date,
			Pill_Misc_Name.URL,
			Pill_Misc_Name.Name,
			Pill_Misc_Name.Other_Name,
			Pill_Misc_Name.Dose
		FROM
			Pill_Misc_Name, Location
		WHERE
			Pill_Misc_Name.Location_Name = Location.Location_Name
		;""")

	# VIEW SOM_Data
	# Pill_ID | Date | X | Y | Z | MDMA_Content | Enactogen_Content | Psychedelic_Content
	# | Cannabinoid_Content | Dissociative_Content | Stimulant_Content | Depressant_Content
	# | Other
	c.execute("""CREATE VIEW Pill_Classification AS
		SELECT
		    Pill_Misc.Pill_ID,
		    Pill_Misc.Date,
		    Location.X,
		    Location.Y,
		    Location.Z,
		    SUM(Pill_Content.Substance_Percentage * Substance.MDMA ) as MDMA_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Enactogen) as Enactogen_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Psychedelic) as Psychedelic_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Cannabinoid ) as Cannabinoid_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Dissociative) as Dissociative_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Stimulant ) as Stimulant_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Depressant) as Depressant_Content,
		    SUM(Pill_Content.Substance_Percentage * Substance.Other) as Other_Content
		FROM
		    Pill_Misc, Pill_Content, Substance, Location, Pill_Misc
		WHERE Pill_Misc.Pill_ID = Pill_Content.Pill_ID
		AND Pill_Misc.Location_ID = Location.Location_ID
		AND Pill_Content.Substance_ID = Substances.Substance_ID
		GROUP BY Pill.Pill_ID
	;""")

	conn.commit()
	conn.close()

