"""
The main program for execution.

Project: EcstasyData
Path: root/main.py
"""
import get_data

import classifications as clss

import sqlite3


get_data.create_database('https://www.ecstasydata.org/search.php?source=1&Max=5000&style=data_only', 'EcstasyData.db')