"""
This program fetches data from ecstacy data.org

Project: EcstasyData
Path: root/get_data.py
"""

import classifications as clss
import visualizer as vis
import exceptions

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

import US_States

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'
warnings.formatwarning = custom_formatwarning

def get_coordinates(location_str, no_query=False, replace_US_states = True):
	"""
	Finds the x, y, z coordinates of the unit vector to a city on the globe.

	Args:
		location_str: A string describing the location.
		no_query: Boolean, Optional False
			If true, makes the function return 0,0,1,0,0 with no wait time and
			no querys to the geocoder.
		replace_US_states: Boolean, Optional True
			Replace all *uppercase** state abbreviations with the full name.

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
		raise exceptions.LocationError(location_str + ' is empty')

	# Create a client to query.
	with open('API\\TomTom') as f:
		tomtomAPI = f.readlines()[0].rstrip()
	# geolocator = geopy.geocoders.TomTom(tomtomAPI)
	geolocator = geopy.geocoders.Nominatim(user_agent='Ecstasy_Data_2018', timeout=4)

	# Geocoding an address
	try:
		location = geolocator.geocode(loc_query)

		# Retry with US state abbreviations changed to full state names
		if location == None and replace_US_states:
			for abbr, full in US_States.states.items():
				loc_query = loc_query.replace(abbr, full)
			time.sleep(2)
			location = geolocator.geocode(loc_query)

	except geopy.exc.GeocoderTimedOut:
		raise exceptions.LocationError(location_str + ' interpreted as ' + loc_query)

	if location == None:
		raise exceptions.LocationError(location_str)
	xyz = vis.lat_lng_to_x_y_z(location.latitude, location.longitude)
	x, y, z = xyz
	return location.latitude, location.longitude, x, y, z

class EDataDB:
	# Class for interacting with ecstasy data database

	# TABLE Location
	# Location_ID | Location_Name | Latitude | Longitude | X | Y | Z

	# TABLE Source
	# Source_ID | Source_Name

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
		self.bad_locations = dict()
		self.bad_pills = list()
		self.encoding = 'Latin-1'

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
		unknown_data = list(self.unknown_substance_count.items())
		out_f = open(path, 'w')
		for substance, count in sorted(unknown_data, key = lambda x : (x[1],x[0])):
			out_f.write(str(count))
			out_f.write('    ')
			out_f.write(substance)
			out_f.write('\n')
		out_f.close()

	def dump_bad_locations(self, path):
		out_f = open(path, 'w')
		for locale, count in sorted(list(self.bad_locations.items()), key = lambda x : (x[1],x[0])):
			out_f.write(str(count))
			out_f.write('    ')
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
				Latitude REAL,
				Longitude REAL,
				X REAL,
				Y REAL,
				Z REAL
			);
		""")

		# Source
		self.c.execute("""
			CREATE TABLE IF NOT EXISTS Source (
				Source_ID INTEGER PRIMARY KEY,
				Source_Name TEXT UNIQUE
			);
		""")

		# Substance
		self.c.execute("""
			CREATE TABLE IF NOT EXISTS Substance (
				Substance_ID INTEGER PRIMARY KEY,
				Substance_Name TEXT UNIQUE,
				MDMA REAL,
				Enactogen REAL,
				Psychedelic REAL,
				Cannabinoid REAL,
				Dissociative REAL,
				Stimulant REAL,
				Depressant REAL,
				Other REAL
			);
		""")

		# Pill_Content
		self.c.execute("""	
			CREATE TABLE IF NOT EXISTS Pill_Content (
				Pill_ID INTEGER,
				Substance_ID INTEGER,
				Substance_Parts REAL,
				Substance_Percentage REAL
			);
		""")

		# Pill_Misc
		self.c.execute("""
			CREATE TABLE IF NOT EXISTS Pill_Misc (
				Pill_ID INTEGER PRIMARY KEY,
				Location_ID INTEGER,
				Source TEXT,
				Sold_As_Ecstasy INTEGER,
				Date TEXT,
				URL TEXT,
				Name TEXT,
				Other_Name TEXT,
				Dose REAL
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
					AND Pill_Misc.Sold_As_Ecstasy = 1
					AND Pill_Misc.Location_ID = Location.Location_ID
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
		return self.add_location_alias(location_string, location_string)

	def add_location_alias(self, location_string, alias, fix_null = False):
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
				latitude, longitude, x, y, z = get_coordinates(alias)
				self.c.execute("""
					INSERT INTO Location
					VALUES (null, ?, ?, ?, ?, ?, ?);
				""", (location_string, latitude, longitude, x, y, z))
			except exceptions.LocationError as e:
				if location_string not in self.bad_locations.keys():
					self.bad_locations[location_string] = 0
				self.bad_locations[location_string] += 1
				self.c.execute("""
					INSERT INTO Location
					VALUES (null, ?, null, null, null, null, null);
				""", (location_string,))
			# Get the key. It's the old maximum key plus one.
			return max(all_IDs) + 1
		else:
			# Get the line from the location table
			self.c.execute('SELECT * FROM Location WHERE Location_Name = ?', (location_string,))
			location_ID, name, lat, lon, X, Y, Z = self.c.fetchone()
			# Check if the line has null values
			if fix_null and (lat == None or lon == None or X == None or Y == None or Z == None):
				# Look for coordinates
				try:
					latitude, longitude, x, y, z = get_coordinates(alias)
					self.c.execute("""
						UPDATE Location
						SET
							Latitude = ?,
							Longitude = ?,
							X = ?,
							Y = ?,
							Z = ?
						WHERE
							Location_ID == ?;
					""", (latitude, longitude, x, y, z, location_ID))
				except exceptions.LocationError as e:
					pass
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
			except exceptions.ClassificationError:
				raise exceptions.ClassificationError(substance_string)
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

	def add_source(self, source):
		# Check if the source is in the database
		self.c.execute('SELECT * FROM Source')
		source_table = self.c.fetchall()
		if len(source_table) == 0:
			sources = []
			source_IDs = [0]
		else:
			source_IDs, sources = zip(*source_table)
		if source in sources:
			return source_IDs[sources.index(source)]
		else:
			new_ID = max(source_IDs) + 1
			self.c.execute("""
				INSERT INTO Source VALUES (?, ?);
			""", (new_ID, source))
			return new_ID

	def load_html_line(self, line):
		"""
		Parses data in the form,
		DataDataID|URL|ThumbnailURL|DetailImage1|ReagentImage1|Name|OtherName|SubmitterDigitCode|SoldAsEcstasy|
		Substance (sep by ;;)|DatePublished|DateTested (approx)|LocationString|SizeString|DataSource
		and adds it to the database.
		"""
		raw_tuple = line.split('|')

		pill_ID = int(raw_tuple[0])

		# Check if this pill has already been added to the database.
		self.c.execute('SELECT Pill_ID FROM Pill_Misc')
		if (pill_ID,) in self.c.fetchall():
			return

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

		source = raw_tuple[14]

		substance_str = raw_tuple[9]
		composition = self._parse_substance_str(substance_str)

		location_str = raw_tuple[12]

		# Begin with TABLE Pill_Misc. First we need to add the location
		location_ID = self.add_location(location_str)

		# Also add the source of the data
		source_ID = self.add_source(source)

		# Now add a row to Pill_Misc
		dose_format = '?'
		if dose == 0: dose_format = 'null'
		pill_misc_format = '(?, ?, ?, %s, ?, ?, ?, ?, %s)' % (['?', '?', 'null'][sold_as_ecstasy], dose_format)
		pill_misc_data = [pill_ID, location_ID, source_ID]
		if sold_as_ecstasy != 2:
			pill_misc_data.append(sold_as_ecstasy)
		pill_misc_data += [date, URL, name, other_name]
		if dose != 0:
			pill_misc_data.append(dose)
		pill_misc_data = tuple(pill_misc_data)
		# print('INSERT INTO Pill_Misc VALUES '+pill_misc_format+';', pill_misc_data)
		# exit()
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
			raise exceptions.ClassificationError(substance)
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
		if dose_str in ['', '-']:
			return 0
		# Get the numeric part
		try:
			number_tag = re.compile('[0-9]+\.*[0-9]* mg')
			numeric = number_tag.search(dose_str).group()
			# Remove the mg suffix (this is assumed)
			mg_dose = float(numeric.rstrip(' mg'))
			return mg_dose
		except AttributeError:
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
		if substance_str in clss.aliases_for_nothing:
			raise exceptions.TestingError(substance_str)
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
				part = part.rstrip('mg').rstrip('ug').rstrip()
				if part == '':
					part = 0
				else:
					try:
						part = float(part)
					except:
						raise exceptions.SubstanceError(substance_str)
					substance_parts.append(part)

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

	def _read_url(self, url):
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
		html = html_bytes.decode(self.encoding)
		uf.close()

		start_flag = 'BEGIN_DATA_BLOCK'
		end_flag = 'END_DATA_BLOCK'
		start_index = html.find(start_flag)
		end_index = html.find(end_flag)
		no_header_footer = html[start_index+len(start_flag) : end_index]
		data_string = no_header_footer.replace('<br>\n', '')
		data_string = data_string.rstrip().lstrip()

		# Get rid of blank lines
		data_lines = data_string.splitlines()
		data_lines = [line for line in data_lines[1:] if line.rstrip() != '']

		return data_lines

	def load_data_only_url(self, url):
		"""
		Loads data from an EcstasyData webpage that needs no url formatting.
		"""
		data_lines = self._read_url(url)

		# Add each data line to the database
		bar = progressbar.ProgressBar(max_value=len(data_lines), redirect_stdout=False).start()
		for line_no, line in enumerate(data_lines):
			bar.update(line_no)
			try:
				self.load_html_line(line)
			except exceptions.TestingError:
				self.bad_pills.append(int(line.split('|')[0]))
			except exceptions.SubstanceError:
				self.bad_pills.append(int(line.split('|')[0]))
		bar.finish()
		print('{}/{} pills failed to be added ({:.1f}%)'.format(\
			len(self.bad_pills), len(data_lines), 100*len(self.bad_pills)/len(data_lines)))

	def load_url(self, url):
		"""
		Loads data from an EcstasyData webpage. E.g. Loads from https://www.ecstasydata.org/results.php.
		"""
		# First we need to add the right things to the url, i.e. &Max=5000&style=data_only
		# but max needs to be high enough to hold all the data.
		# FINDING OUT HOW MANY DATA POINTS THERE ARE:
		# First load the url
		with urllib.request.urlopen(url) as response:
			html_bytes = response.read()
		html = html_bytes.decode(self.encoding)

		# Look for <li class="Results">_____ entries total</li>. The underscores are the location of the
		li_tag = re.compile('<li class="Results">[0-9]+ entries total</li>')
		matching_tag = li_tag.search(html).group()
		# total number of data points.
		integer_re = re.compile('[0-9]+')
		n_data = integer_re.search(matching_tag).group()
		# Add the proper flags to the url and load
		print(url+'?&Max=%s&style=data_only' % (n_data,))
		self.load_data_only_url(url+'?&Max=%s&style=data_only' % (n_data,))

	def __enter__(self):
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		self.close()