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
import visualizer as vis

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
        neighborhood_function='gaussian', random_seed=None, ignore_date = True, distance_weight=1,
        normalization_mode = 'None', ID = False):

        self.distance_weight = distance_weight

        self.ID = ID
        if self.ID:
            self.data_raw = [x[1:] for x in data_list]
            self.IDs = [x[0] for x in data_list]
        else:
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

        self.ignore_date = ignore_date

        self.features = len(self.data_raw[0]) - ignore_date

        super(ClassifiedSOM, self).__init__(x, y, self.features, sigma=sigma,
            learning_rate=learning_rate, decay_function = decay_function,
            neighborhood_function=neighborhood_function, random_seed=random_seed)

        self.data = [x[1:] for x in self.data_raw]
        if ignore_date:
            self.data = [x[1:] for x in self.data_raw]
        else:
            # Need to convert all dates to ordinals
            self.data = map(lambda x: [x[0].toordinal()] + x[1:], self.data)
            # all_dates = [x[0] for x in self.data]
            # start_date = min(all_dates)
            # end_date = max(all_dates)
            # def __normalize_date(date_ordinal):
            #     return (date_ordinal - start_date)/(end_date-start_date)
            # self.data = map(lambda x : [__normalize_date(x[0])] + x[1:], self.data)
        self.data = np.array(self.data)

        if normalization_mode == 'None':
            self.data_rescale = np.ones(self.data[0].shape)
            self.data_mean = np.zeros(self.data[0].shape)
        if normalization_mode == 'Gaussian':
            # Create Gaussian normalized data.
            self.data_rescale = np.std(self.data, axis = 0)
            # Find the mean along every feature
            self.data_mean = np.average(self.data, axis = 0)
        if normalization_mode == 'Linear':
            # Create Gaussian normalized data.
            self.data_rescale = np.amax(self.data, axis = 0) - np.amin(self.data, axis = 0)
            # Find the mean along every feature
            self.data_mean = np.average(self.data, axis = 0)
        if normalization_mode == 'LinearAbs':
            # Create Gaussian normalized data.
            self.data_rescale = np.amax(self.data, axis = 0) - np.amin(self.data, axis = 0)
            # Find the mean along every feature
            self.data_mean = np.amin(self.data, axis = 0)

        self.data_n = self.normalize(self.data)

        self.dist_map = None
        self.act_resp = None
        self.clear_clusters()

    def normalize(self, vector, uncertainty = False):
        """
        Normalizes data for the SOM.

        Args:
            vector : numpy float64 ndarray
                Data from the table.
            uncertainty : boolean, default False
                If true, treats the vector as an uncertainty (does not subtract mean).

        Returns:
            0 : numpy float64 ndarray
            Data from the SOM.
        """
        if uncertainty:
            weight = vector/self.data_rescale
        else:
            weight = (vector - self.data_mean) / self.data_rescale
        return weight

    def denormalize(self, weight, uncertainty = False):
        """
        Deormalizes data for the SOM.

        Args:
            weight : numpy float64 ndarray
                Data from the SOM.
            uncertainty : boolean, default False
                If true, treats the vector as an uncertainty (does not subtract mean).

        Returns:
            0 : numpy float64 ndarray
                Data from the table.
        """
        if uncertainty:
            vector = weight * self.data_rescale
        else:
            vector = weight * self.data_rescale + self.data_mean
        return vector

    def train(self, num_iteration):
        """
        Train the SOM.

        Args:
            N : Integer
                The number of iterations to train for.
        """
        """Trains the SOM picking samples at random from data"""
        self._check_iteration_number(num_iteration)
        self._check_input_len(self.data_n)

        bar = progressbar.ProgressBar(max_value = num_iteration, redirect_stdout=True).start()
        for iteration in range(num_iteration):
            # pick a random sample
            rand_i = self._random_generator.randint(len(self.data_n))
            self.update(self.data_n[rand_i], self.winner(self.data_n[rand_i]),
                        iteration, num_iteration)
            bar.update(iteration)
        bar.finish()

    def generate_distance_map(self):
        self.dist_map = self.distance_map()

    def generate_activation_response(self):
        self.act_resp = self.activation_response(self.data_n)

    def plot_distance_map(self, path=None):
        if self.dist_map is None:
            self.generate_distance_map()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.dist_map, interpolation='spline16')
        fig.colorbar(cax, orientation='horizontal')
        plt.title('Distance Map', y=1.1)

        if path != None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def plot_activation_response(self, path=None):
        if self.act_resp is None:
            self.generate_activation_response()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_max = 10**np.ceil(np.log10(self.act_resp.max()))
        cax = ax.matshow(self.act_resp, interpolation='spline16', norm=LogNorm(vmin=1, vmax=plot_max))
        fig.colorbar(cax, orientation='horizontal')
        plt.title('Activation Response', y=1.1)
        
        if path != None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def clear_clusters(self):
        self.cluster_map = -np.ones(self.shape, dtype = int)
        self.cluster_map_normalized = -np.ones(self.shape)
        self.cluster_weights = np.zeros(self.shape + (self.features,))
        self.cluster_std = np.zeros(self.shape + (self.features,))
        self.clusters = []

    def plot_clusters(self, path=None, normalization = 'unnormalized'):
        """
            Plots the cluster map.

            Args:
                path : string, default None
                    Save path. If None, just displays the plot.
                normalization : string, default 'unnormalized'
                    Normalization to apply to the cluster map. If 'unnormalized', just
                    plots the activation total of the cluster. If 'size', normalizes
                    to the size of the cluster.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if normalization == 'unnormalized':
            to_plot = np.ma.masked_where(self.cluster_map < 0, self.cluster_map)
        elif normalization == 'size':
            to_plot = np.ma.masked_where(self.cluster_map < 0, self.cluster_map_normalized)

        cax = ax.matshow(to_plot, interpolation='nearest', vmin = 0)
        cb = fig.colorbar(cax, orientation='horizontal')
        if normalization == 'unnormalized':
            cb.set_label('Cluster Activation', labelpad = 15)
        if normalization == 'size':
            cb.set_label('Normalized Activation', labelpad = 15)
        plt.title('Cluster Map', y=1.1)

        if path is not None:
            plt.savefig(path)
            plt.close()
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
                The average weight of the cluster.
            1 : float, numpy array
                The standard deviation of the average weight calculation.
            2 : integer
                The number of data points in the cluster.
            3 : integer list of 2-tuples
                All the indeces included in the cluster.
        """
        if cluster_indeces is None:
            cluster_indeces = self.cluster_indeces(center_index, threshold)

        if self.act_resp is None:
            self.generate_activation_response()


        ravelled_indeces = np.ravel_multi_index(np.array(cluster_indeces).transpose(), self.shape)

        activations = self.act_resp.take(ravelled_indeces)
        total_activation = np.sum(activations)

        cluster_weights = []
        for index in cluster_indeces:
            cluster_weights.append(self[index])
        cluster_weights = np.array(cluster_weights)
        try:
            average_weight = np.average(cluster_weights, axis = 0, weights = activations)
        except ZeroDivisionError:
            # There is no activation of this cluster
            raise exceptions.ClusterError('Inactive cluster at ' + str(center_index))
        weight_std = np.sqrt(np.average((cluster_weights - average_weight)**2, axis = 0, weights = activations))

        # unit_sum = 0
        # for index in cluster_indeces:
        #     unit_sum += self.act_resp[index] * self[index]

        # try:
        #     average_weight = unit_sum/total_activation
        # except ZeroDivisionError:
        #     # There is no activation of this cluster
        #     raise exceptions.ClusterError('Inactive cluster at ' + str(center_index))
        return average_weight, weight_std, total_activation, list(cluster_indeces)

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

    def add_cluster(self, average_weight, weight_std, total_activation, cluster_indeces):
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
        normal_activation = total_activation / len(cluster_indeces)
        for index in cluster_indeces:
            # Overlapping clusters!
            if self.cluster_map[index] != -1:
                raise exceptions.ClusterError('Clusters overlapping at ' + str(index) + str(self.cluster_map[index]))
            else:
                self.cluster_map[index] = total_activation
                self.cluster_map_normalized[index] = normal_activation
                self.cluster_weights[index] = average_weight.transpose()
                self.cluster_std[index] = weight_std.transpose()

        self.clusters.append(cluster_indeces)

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
            self.cluster_map_normalized[index] = -1
            self.cluster_weights[index].fill(0) 
            self.cluster_std[index].fill(0) 

    def cluster(self, threshold):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                center_index = (i, j)
                # If this cluster already exists,
                # its center_index will already be in a cluster.
                # Thus, we only need to check  the center index.
                if self.cluster_map[center_index] == -1:
                    # This cluster doesn't already exist. Get the indeces
                    cluster_indeces = self.cluster_indeces(center_index, threshold)
                    if len(cluster_indeces) != 0:
                        try:
                            average_weight, weight_std, total_activation, cluster_indeces_list = \
                            self.cluster_weight(center_index, threshold, cluster_indeces=cluster_indeces)
                            self.add_cluster(average_weight, weight_std, total_activation, cluster_indeces_list)
                        except exceptions.ClusterError:
                            # We ran into an inactive cluster.
                            pass

    def cluster_analysis(self):
        """
        Finds the center of every cluster, gives the number of members, the average weight, and the weight standard deviation.

        Returns:
            0 : list
                Each element takes the form (cluster_center, average_weight, std)
                where cluster center is an integer 2-tuple, average weight is a numpy
                1d array, and std is a numpy 1d array.
        """
        analysis = []
        for cluster_indeces in self.clusters:
            cluster_center = self.find_center(cluster_indeces)
            cluster_size = len(cluster_indeces)
            activations = self.cluster_map[cluster_center]
            normalized_activations = float(activations)/cluster_size
            average_weight = self.cluster_weights[cluster_center]
            std = self.cluster_std[cluster_center]
            analysis.append((activations, cluster_size, normalized_activations, cluster_center, average_weight, std))

        # We want to sort the analysis as well as sort self.clusters
        analysis_clusters = sorted(zip(analysis, self.clusters), reverse = True)
        analysis, self.clusters = zip(*analysis_clusters)

        return analysis

    def dump_cluster_analysis(self, path):
        """
        Generates and writes out a cluster analysis.

        Args:
            0 : string
                Path to output file.
        """
        with open(path, 'w') as out_f:
            for activations, size, normalized_activations, center, weight, std in self.cluster_analysis():
                out_f.write(str(activations))
                out_f.write('    ')
                out_f.write(str(size))
                out_f.write('    ')
                out_f.write(str(normalized_activations))
                out_f.write('    ')
                out_f.write(str(center))
                out_f.write('    ')
                out_f.write(repr(weight).replace('\n', ' '))
                out_f.write('    ')
                out_f.write(repr(std).replace('\n', ' '))
                out_f.write('\n')

    def cluster_report(self, path):
        """
        Generates and writes out a cluster analysis in a human readable form.

        Args:
            0 : string
                Path to output file.
        """
        def zip_error(values, uncertainties):
            for v, u in zip(values, uncertainties):
                v = v.round(2)
                u = u.round(2)
                yield '%7.2f(%4.2f)' % (v, u)

        # def zip_error_latlng(latlng, latlngerr):
        #     for v, u in zip(latlng, latlngerr):
        #         v = v.round(2)
        #         u = u.round(2)
        #         yield '%6.2f(%5.2f)' % (v, u)

        with open(path, 'w') as out_f:
            out_f.write('#Activations    Cluster Size   Act/Size   i   j       MDMA          Enactogen     Psychedelic   Cannabinoid   Dissociative  Stimulant     Depressant    Other\n')
            for activations, size, normalized_activations, center, weight, std in self.cluster_analysis():
            #     if self.ignore_date:
            #         x = weight[0]
            #         y = weight[1]
            #         z = weight[2]
            #         dx = std[0]
            #         dy = std[1]
            #         dz = std[2]
            #     else:
            #         x = weight[1]
            #         y = weight[2]
            #         z = weight[3]
            #         dx = std[1]
            #         dy = std[2]
            #         dz = std[3]
            #     latlng, latlngerr = vis.x_y_z_to_lat_lng(x, y, z, error = (dx,dy,dz))
                # We want to output vectors, not weights
                vector = self.denormalize(weight)
                std = self.denormalize(std, uncertainty = True)
                out_f.write('%10d    %10d    %8.2f    ' % (activations, size, normalized_activations))
                out_f.write('%3d %3d' % center)
                out_f.write('    ')
                for x in zip_error(vector, std):
                    out_f.write(x)
                    out_f.write(' ')
                # for coord in zip_error_latlng(latlng, latlngerr):
                #     out_f.write(coord)
                #     out_f.write(' ')
                out_f.write('\n')

    def find_center(self, cluster_indeces):
        index_array = np.array(cluster_indeces)
        mean_index = np.average(index_array, axis=0)
        diff = np.sum(np.abs(index_array - mean_index), axis = 1)
        closest_index = diff.argmin()
        return cluster_indeces[closest_index]

    def member_IDs(self, cluster_indeces):
        """
        Returns a list of all the data IDs which are a member of this cluster.
        """
        assert self.ID

        members = []

        for ID, weight in zip(self.IDs, self.data_n):
            # Find out what the winner neuron is
            winner_index = self.winner(weight)
            if winner_index in cluster_indeces:
                members.append(ID)
        return members

    def __getitem__(self, indeces):
        return self._weights.__getitem__(indeces)