"""
This interfaces with MiniSom to apply a self organizing map to the database.

Project: EcstasyData
Path: root/exceptions.py
"""

########## DATA LOADING EXCEPTIONS ###############
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

########## SOM EXCEPTIONS ##############
class ClusterError(Exception):
    """
    Raised for issues dealing with clustering SOMs.
    """
    pass