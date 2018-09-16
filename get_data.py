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

def create_database(url, path):
	html = _load_data(url)
	data_string = _remove_junk(html)
	processed_data = _read_to_list(data_string)
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
		SoldAsEcstasy integer, SubstanceMDMA real, SubstanceEcstasyLike real,
		SubstancePsychedelic real, SubstanceCannabinoid real,
		SubstanceDissociative real, SubstanceStimulant real,
		SubstanceDepressant real, SubstanceOther real, Date text,
		x real, y real, z real, Dose integer), ...]
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
			processed_tuple = _raw_tuple_to_processed_data(raw_tuple, warn_other=True)
			processed_data.append(processed_tuple)
			good_lines += 1
		except ValueError as e:
			bad_lines += 1
			warnings.warn('Could not read the following line\n'+str(raw_tuple)+'\nbecause '+str(e))
		except AttributeError:
			bad_lines += 1
			warnings.warn('Attribute issue with the following line\nbecause'+str(raw_tuple))
		except SubstanceError as substance_str:
			bad_lines += 1
			warnings.warn(str(raw_tuple[0]) + ' Has no substances ' + str(substance_str))
		except TestingError as substance_str:
			untested_lines += 1
			# warnings.warn(str(raw_tuple[0]) + ' was not tested')
		except UnidentifiedError as unknown_substance:
			unknown_chem_lines += 1
			unknown_substances.append(str(unknown_substance))
			warnings.warn(str(raw_tuple[0]) + ' Unknown substance ' + str(unknown_substance) + ' encountered')

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
	percentages = [str(round(100*x/total_lines,1)) for x in (good_lines, bad_lines, unknown_chem_lines, untested_lines)]
	output_figures = tuple([str(total_lines)] + percentages)
	print('Processed %s lines. %s%% kept, %s%% poorly formatted, %s%% with unknown substances, %s%% untested.' %\
		output_figures)
	return processed_data

def _raw_tuple_to_processed_data(raw_tuple, warn_other = False):
	"""
	Interprets a tuple with elements
			DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
	into the form
		(DataDataID integer, URL text, Name text, OtherName text,
		SoldAsEcstasy integer, SubstanceMDMA real, SubstanceEcstasyLike real,
		SubstancePsychedelic real, SubstanceCannabinoid real,
		SubstanceDissociative real, SubstanceStimulant real,
		SubstanceDepressant real, SubstanceOther real, Date text,
		x real, y real, z real, Dose integer)
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
	composition = _parse_substance_str(Substance_str, warn_other = warn_other)

	SubstanceMDMA = composition[0]

	SubstanceEcstasyLike = composition[1]

	SubstancePsychedelic = composition[2]

	SubstanceCannabinoid = composition[3]

	SubstanceDissociative = composition[4]

	SubstanceStimulant = composition[5]

	SubstanceDepressant = composition[6]

	SubstanceOther = composition[7]

	Date_str = raw_tuple[11]
	Date = _parse_date(Date_str)

	Location_str = raw_tuple[12]
	x,y,z = _get_coordinates(Location_str)

	Dose_str = raw_tuple[13]
	Dose = _parse_dose(Dose_str)

	return (DataDataID, URL, Name, OtherName, SoldAsEcstasy, SubstanceMDMA, \
		SubstanceEcstasyLike, SubstancePsychedelic, SubstanceCannabinoid, \
		SubstanceDissociative, SubstanceStimulant, SubstanceDepressant, \
		SubstanceOther, Date, x, y, z, Dose)

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
		0: (x,y,z)
	"""
	# # Create a client to query.
	# geolocator = Nominatim(user_agent='Ecstasy_Data_2018')
	# # Geocoding an address
	# location = geolocator.geocode(location_str)
	# xyz = vis.lat_lng_to_x_y_z(location.latitude, location.longitude)
	xyz = (0,0,0)
	return xyz 

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

def _parse_substance_str(substance_str, warn_other = False):
	"""
	Parses a substance string.

	Args:
		substance_str: The substance string in the html formal
		warn_other: Default False. If true, warns the user every time a
			substance is classified as other.

	Returns:
		0: List of percentages [MDMA, Ecstasy-like substances,
			Psychedelics, Cannabinoids, Dissociatives, Stimulants,
			Depressants, Other]
	"""
	# Deal with untested
	if substance_str in clss.aliases_for_nothing:
		raise TestingError(substance_str)
	# Deal with sugar pills
	if 'None Detected' in substance_str:
		return [0,0,0,0,0,0,0,1]
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
	try:
		substance_percentages = [x/total_parts for x in substance_parts]
	except ZeroDivisionError:
		# This exception triggers when only trace quantities of
		# chemicals are found (i.e. when we only have things with 0s or ---)
		return [0,0,0,0,0,0,0,1]

	# Convert substance names to substance types
	substance_types = [classify_substance(sub_name, warn_other=warn_other) \
		for sub_name in substance_names]

	# Join substance types together
	composition = [0]*9
	for substance_type, substance_percentage in \
		zip(substance_types, substance_percentages):
		composition[substance_type] += substance_percentage

	return composition[:-1]

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

def _list_to_database(path, data_list):
	"""
	Converts a list to a database with sqlite3.

	Args:
		path: Path to store the database.
		data_list: List of processed tuples for data.
	"""
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("""CREATE TABLE edata (
			DataDataID integer,
			URL text,
			Name text,
			OtherName text,
			SoldAsEcstasy integer,
			SubstanceMDMA real,
			SubstanceEcstasyLike real,
			SubstancePsychedelic real,
			SubstanceCannabinoid real,
			SubstanceDissociative real,
			SubstanceStimulant real,
			SubstanceDepressant real,
			SubstanceOther real,
			Date text,
			x real,
			y real,
			z real,
			Dose integer
		)""")

	outF = open('foo.txt', 'w')

	for data_tuple in data_list:
		c.execute('INSERT INTO edata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data_tuple)
		outF.write(str(data_tuple)+'\n')
	outF.close()

	conn.commit()
	conn.close()

