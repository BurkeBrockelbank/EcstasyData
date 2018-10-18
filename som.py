"""
This interfaces with MiniSom to apply a self organizing map to the database.

Project: EcstasyData
Path: root/som.py
"""

import minisom

import numpy as np

class ClassifiedSOM(minisom.MiniSom):
    """
    This object holds data for training the SOM and also has some features for interfacing with it.

    Args:
        data_list: List of tuples of the form:
            [(Pill_ID, Date, X, Y, Z, MDMA, Enactogen, Psychedelic, Cannabinoid, Dissociative,
              Stimulant, Depressant, Other),...]
            containing all the data for pills collected.
        dimensions: int tuple, optional (default=5*sqrt(samples))
            (x, y) dimensions of the SOM (2D)
        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
            learning_rate, initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)
        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/max_iterarations)
            A custom decay function will need to to take in input
            three parameters in the following order:
            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed
            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.
        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat', 'bubble'
        random_seed : int, optional (default=None)
            Random seed to use.
    """
    def __init__(self, data_list, dimensions=None, sigma=1.0, learning_rate=0.5,
        decay_function = None,
        neighborhood_function='gaussian', random_seed=None, ignore_date = True):

        self.data_raw = data_raw
        if dimensions == None:
            samples = len(self.data_list)
            neurons = 5*np.sqrt(samples)
            width = int(np.ceil(np.sqrt(neurons)))
            x = width
            y = width
        else:
            x, y = dimensions

        super(ClassifiedSOM, self).__init__(x, y, 12, sigma=sigma,
            learning_rate=learning_rate, decay_function = None,
            neighborhood_function=neighborhood_function, random_seed=random_seed)

        self.data = [x[1:] for x in self.data_raw]
        if ignore_date:
            self.data = [x[1:] for x in self.data_raw]
        else:
            # Need to normalize all the dates. First change everything to ordinals
            self.data = map(lambda x: [x[0].toordinal()] + x[1:], self.data)
            all_dates = [x[0] for x in self.data]
            start_date = min(all_dates)
            end_date = max(all_dates)
            def __normalize_date(date_ordinal):
                return (date_ordinal - start_date)/(end_date-start_date)
            self.data = map(lambda x : [__normalize_date(x[0])] + x[1:], self.data)
        self.data = np.array(self.data)

    def train(N):
        """
        Train the SOM.

        Args:
            N : Integer
                The number of iterations to train for.
        """
        self.train_random(self.data, N)
