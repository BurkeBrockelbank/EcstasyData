"""
This interfaces with MiniSom to apply a self organizing map to the database.

Project: EcstasyData
Path: root/som.py
"""

import minisom

import numpy as np

import progressbar

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import exceptions

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
        decay_function = minisom.asymptotic_decay,
        neighborhood_function='gaussian', random_seed=None, ignore_date = True):

        self.data_raw = data_list
        if dimensions == None:
            samples = len(self.data_raw)
            neurons = 5*np.sqrt(samples)
            width = int(np.ceil(np.sqrt(neurons)))
            x = width
            y = width
        else:
            x, y = dimensions

        self.shape = (x,y)

        self.features = 12 - ignore_date

        super(ClassifiedSOM, self).__init__(x, y, self.features, sigma=sigma,
            learning_rate=learning_rate, decay_function = decay_function,
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

        self.dist_map = None
        self.act_resp = None
        self.cluster_map = -np.ones(self.shape, dtype = int)
        self.cluster_weights = np.zeros(self.shape + (1,)*self.features)

    def train(self, num_iteration):
        """
        Train the SOM.

        Args:
            N : Integer
                The number of iterations to train for.
        """
        """Trains the SOM picking samples at random from data"""
        self._check_iteration_number(num_iteration)
        self._check_input_len(self.data)

        bar = progressbar.ProgressBar(max_value = num_iteration, redirect_stdout=True).start()
        for iteration in range(num_iteration):
            # pick a random sample
            rand_i = self._random_generator.randint(len(self.data))
            self.update(self.data[rand_i], self.winner(self.data[rand_i]),
                        iteration, num_iteration)
            bar.update(iteration)
        bar.finish()

    def generate_distance_map(self):
        self.dist_map = self.distance_map()

    def generate_activation_response(self):
        self.act_resp = self.activation_response(self.data)

    def plot_distance_map(self, path=None):
        if self.dist_map is None:
            self.generate_distance_map()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.dist_map, interpolation='nearest')
        fig.colorbar(cax)
        plt.title('Distance Map')

        if path != None:
            plt.savefig(path)
        else:
            plt.show()

    def plot_activation_response(self, path=None):
        if self.act_resp is None:
            self.generate_activation_response()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_max = 10**np.ceil(np.log10(self.act_resp.max()))
        cax = ax.matshow(self.act_resp, interpolation='nearest', norm=LogNorm(vmin=1, vmax=plot_max))
        fig.colorbar(cax)
        plt.title('Activation Response')
        
        if path != None:
            plt.savefig(path)
        else:
            plt.show()

    def plot_clusters(self, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.cluster_map, interpolation='nearest', vmin = 0)
        fig.colorbar(cax)
        plt.title('Distance Map')

        if path != None:
            plt.savefig(path)
        else:
            plt.show()

    def cluster_indeces(self, center_index, threshold):
        """
        Finds the indeces of a cluster given the center index and the threshold.

        Args:
            center_index : integer 2-tuple
                Index for the center of the cluster search.
            threshold : float
                Distance threshold for inclusion in the cluster.
        """
        if self.dist_map is None:
            self.generate_distance_map()

        cluster_indeces = set([])
        unchecked_indeces = set([center_index])
        while len(unchecked_indeces) > 0:
            new_unchecked_indeces = set([])

            for index in unchecked_indeces:
                if self.dist_map[index] <= threshold:
                    cluster_indeces.add(index)
                    for neighbor_index in self.__neighbors(index):
                        if neighbor_index not in cluster_indeces:
                            new_unchecked_indeces.add(neighbor_index)

            unchecked_indeces = new_unchecked_indeces

        return list(cluster_indeces)

    def cluster_weight(self, center_index, threshold, cluster_indeces = None):
        """
        This function takes in a index for a cluster center unit and finds all the units closeby
        (maximum step of threshold away in distance map). It then does a weighted average on all
        these values (consulting the activation map) and averages the value of all these
        to get the cluster weight and also returns all the indeces it added to the cluster.

        Args:
            center_index : integer 2-tuple
                Index for the center of the cluster search.
            threshold : float
                Distance threshold for inclusion in the cluster.

        Returns:
            0 : float, numpy array
                The average weight of the cluster
            1 : integer
                The number of data points in the cluster
            2 : integer list of 2-tuples
                All the indeces included in the cluster.
        """
        if cluster_indeces is None:
            cluster_indeces = self.cluster_indeces(center_index, threshold)

        if self.act_resp is None:
            self.generate_activation_response()

        total_activation = 0
        unit_sum = 0
        for index in cluster_indeces:
            total_activation += self.act_resp[index]
            unit_sum += self.act_resp[index] * self[index]

        average_weight = unit_sum/total_activation

        return average_weight, total_activation, list(cluster_indeces)

    def __neighbors(self, index):
        i = index[0]
        j = index[1]
        neighbors = []
        for ip in [-1, 0, 1]:
            for jp in [-1, 0, 1]:
                if ip != 0 or jp != 0:
                    neighbor = (i+ip, j+jp)
                    # If the neighbor is in bounds, add it to the list of neighbors
                    if not(neighbor[0] < 0 or neighbor[1] < 0 or \
                        neighbor[0] >= self.shape[0] or \
                        neighbor[1] >= self.shape[1]):
                        neighbors.append(neighbor)
        return neighbors

    def add_cluster(self, average_weight, total_activation, cluster_indeces):
        """
        Puts a cluster into the cluster map.

        Args:
            average_weight : float, numpy array
                The average weight of the cluster
            total_activation : integer
                The number of data points in the cluster
            cluster_indeces : integer list of 2-tuples
                All the indeces included in the cluster.
        """
        for index in cluster_indeces:
            # Overlapping clusters!
            if self.cluster_map[index] != -1:
                raise exceptions.ClusterError('Clusters overlapping at ' + str(index) + str(self.cluster_map[index]))
            else:
                self.cluster_map[index] = total_activation
                self.cluster_weights[index] = average_weight

    def remove_cluster(self, center_index, threshold, cluster_indeces = None):
        """
        Removes the cluster containing index.

        Args:
            center_index : integer 2-tuple
                One index in the cluster you want removed.
        """
        if cluster_indeces is None:
            cluster_indeces = self.cluster_indeces(center_index, threshold)

        for index in cluster_indeces:
            self.cluster_map[index] = -1
            self.cluster_weights_index[index].fill(0) 

    def cluster(self, threshold):
        for i in self.shape[0]:
            for j in self.shape[1]:
                center_index = (i, j)
                # If this cluster already exists,
                # its center_index will already be in a cluster.
                # Thus, we only need to check  the center index.
                if self.cluster_map[center_index] == -1:
                    # This cluster doesn't already exist
                    self.add_cluster(*self.cluster_weight(center_index, threshold))

    def __getitem__(self, indeces):
        return self._weights.__getitem__(indeces)
