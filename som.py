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

class SOM(minisom.MiniSom):
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
        neighborhood_function='gaussian', random_seed=None,
        normalization_mode = 'None', ID = True, outlier_sigma = None,
        cyclic = None, cyclic_norms = None, cyclic_modes = None):

        self.ID = ID
        if self.ID:
            self.raw_data = np.array([x[1:] for x in data_list])
            self.IDs = np.array([x[0] for x in data_list])
        else:
            self.raw_data = np.array(data_list)
            self.IDs = np.zeros(len(data_list))

        # Deal with cyclic data
        self.cyclic = cyclic
        self.cyclic_norms = cyclic_norms
        self.cyclic_modes = cyclic_modes
        if cyclic is None:
            self.data = self.raw_data
        else:
            assert len(cyclic) == self.raw_data.shape[1]
            assert len(cyclic_norms) == self.raw_data.shape[1]
            assert len(cyclic_modes) == self.raw_data.shape[1]
            # Count number of cyclics
            n_cyc = sum([int(b) for b in cyclic])
            # Create array for new data
            self.data = np.empty((self.raw_data.shape[0], self.raw_data.shape[1]+n_cyc))
            # Fill array
            cyclic_feature_index = 0
            for feature_index, is_cyclic in enumerate(cyclic):
                if is_cyclic:
                    x, y = self.cyclify(self.raw_data[:,feature_index],
                        cyclic_norms[feature_index], cyclic_modes[feature_index])
                    self.data[:,cyclic_feature_index] = x
                    self.data[:,cyclic_feature_index+1] = y
                    cyclic_feature_index += 2
                else:
                    self.data[:,cyclic_feature_index] = self.raw_data[:,feature_index]
                    cyclic_feature_index += 1

        if dimensions == None:
            samples = self.data.shape[0]
            neurons = 5*np.sqrt(samples)
            width = int(np.ceil(np.sqrt(neurons)))
            x = width
            y = width
        else:
            x, y = dimensions

        self.shape = (x,y)

        self.features = self.data.shape[1]

        super().__init__(x, y, self.features, sigma=sigma,
            learning_rate=learning_rate, decay_function = decay_function,
            neighborhood_function=neighborhood_function, random_seed=random_seed)

        if outlier_sigma is not None:
            outliers = len(self.remove_outliers(outlier_sigma))
            print('Removed %d outliers from %d points leaving %d remaining.' % (
                outliers, outliers+self.data.shape[0], self.data.shape[0]))

        self.normalization_init(normalization_mode)
        self.data_n = self.normalize(self.data)

        self.dist_map = None
        self.act_resp = None
        self.clear_clusters()
        self.winners = None

    def cyclify(self, v, norm, mode = 'rad'):
        if mode == 'deg':
            norm = norm/180*np.pi
        elif mode == 'rad':
            pass
        else:
            raise ValueError('Argument mode to cyclify must be \'rad\' or \'deg\'.')
        x = np.cos(v/norm)
        y = np.sin(v/norm)
        return x, y
    
    def decyclify(self, x, y, norm, mode = 'rad'):
        if mode == 'deg':
            norm = norm/180*np.pi
        elif mode == 'rad':
            pass
        else:
            raise ValueError('Argument mode to cyclify must be \'rad\' or \'deg\'.')
        v = norm * np.atan2(y, x)
        return v

    def normalization_init(self, normalization_mode):
        """
        Sets up normalization constants in self.data_rescale and self.data_offset.
        """
        if normalization_mode == 'None':
            self.data_rescale = np.ones(self.data[0].shape)
            self.data_offset = np.zeros(self.data[0].shape)
        elif normalization_mode == 'Gaussian':
            # Create Gaussian normalized data.
            self.data_rescale = np.std(self.data, axis = 0)
            # Find the mean along every feature
            self.data_offset = np.average(self.data, axis = 0)
        elif normalization_mode == 'Linear':
            # Create Gaussian normalized data.
            self.data_rescale = np.amax(self.data, axis = 0) - np.amin(self.data, axis = 0)
            # Find the mean along every feature
            self.data_offset = np.average(self.data, axis = 0)
        elif normalization_mode == 'LinearAbs':
            # Create Gaussian normalized data.
            self.data_rescale = np.amax(self.data, axis = 0) - np.amin(self.data, axis = 0)
            # Find the mean along every feature
            self.data_offset = np.amin(self.data, axis = 0)
        else:
            # Normalization mde incorrectly specified
            raise ValueError('Incorrectly specified normalization mode %s.' % (str(normalization_mode),))

    def remove_outliers(self, sigma):
        """
        Removes outliers from self.data,
        and self.IDs.
        """
        # First get the mean and error range
        mean = np.average(self.data, axis = 0)
        err = np.std(self.data, axis = 0) * sigma
        # Get the maximum and minimum cutoff values
        max_vector = mean + err
        min_vector = mean - err
        # Report
        print('Removing vectors greater than')
        print(max_vector)
        print('and less than')
        print(min_vector)
        # Now iterate through the data
        outlier_indeces = []
        for i, vector in enumerate(self.data):
            if any(np.logical_or(
                vector > max_vector,
                vector < min_vector)):
                # This is a bad index because at least one value
                # in the vector is outside of the acceptable range.
                outlier_indeces.append(i)
        # Delete data points (iterate backwards so indexing isn't screwed up)
        self.data = np.delete(self.data, outlier_indeces, axis = 0)
        self.IDs = np.delete(self.IDs, outlier_indeces, axis = 0)
        return outlier_indeces

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
            weight = (vector - self.data_offset) / self.data_rescale
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
            vector = weight * self.data_rescale + self.data_offset
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

    def plot_distance_map(self, path=None, interpolation = None):
        if self.dist_map is None:
            self.generate_distance_map()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.dist_map, interpolation = interpolation)
        fig.colorbar(cax, orientation='horizontal')
        plt.title('Distance Map', y=1.1)

        if path != None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def plot_activation_response(self, path=None, interpolation = None):
        if self.act_resp is None:
            self.generate_activation_response()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_max = 10**np.ceil(np.log10(self.act_resp.max()))
        cax = ax.matshow(self.act_resp, interpolation=interpolation, norm=LogNorm(vmin=1, vmax=plot_max))
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
        self.members = []

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

        return self.__cluster_indeces(self.dist_map, center_index, threshold)

    def __cluster_indeces(self, array, center_index, threshold):
        """
        Finds the indeces of a cluster given the center index and the threshold.
        """
        cluster_indeces = set([])
        unchecked_indeces = set([center_index])
        while len(unchecked_indeces) > 0:
            new_unchecked_indeces = set([])

            for index in unchecked_indeces:
                if array[index] <= threshold:
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

        for index in cluster_indeces:        
            for cluster in self.clusters:
                if index in cluster:
                    del self.members[index]
                    del self.clusters[index]
                    return

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

    def build_cluster_members(self):
        self.members = []
        for cluster_indeces in self.clusters:
            # Get members
            members = [i for i, x in enumerate(self.winners) if x in cluster_indeces]
            self.members.append(members)

    def build_winners(self):
        """
        Populates self.winners
        """
        bar = progressbar.ProgressBar(max_value = len(self.data_n), redirect_stdout=True).start()
        self.winners = []
        for i, datum in enumerate(self.data_n):
            self.winners.append(self.winner(datum))
            bar.update(i)
        bar.finish()

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
        for cluster_number, cluster_indeces in enumerate(self.clusters):
            cluster_center = self.find_center(cluster_indeces)
            cluster_size = len(cluster_indeces)
            activations = self.cluster_map[cluster_center]
            normalized_activations = float(activations)/cluster_size
            # Now we need the average weight and standard deviation of weights
            member_vectors = np.take(self.raw_data, self.members[cluster_number], axis = 0)
            average_vector = np.average(member_vectors, axis = 0)
            std = np.std(member_vectors, axis = 0)
            # Add this to the analysis
            analysis.append((activations, cluster_size, normalized_activations, cluster_center, average_vector, std))

        # We want to sort the analysis as well as sort self.clusters
        analysis_clusters = sorted(zip(analysis, self.clusters, self.members), reverse = True)
        analysis, self.clusters, self.members = zip(*analysis_clusters)

        return analysis

    def dump_cluster_analysis(self, path):
        """
        Generates and writes out a cluster analysis.

        Args:
            0 : string
                Path to output file.
        """
        with open(path, 'w') as out_f:
            for activations, size, normalized_activations, center, vector, std in self.cluster_analysis():
                out_f.write(str(activations))
                out_f.write('    ')
                out_f.write(str(size))
                out_f.write('    ')
                out_f.write(str(normalized_activations))
                out_f.write('    ')
                out_f.write(str(center))
                out_f.write('    ')
                out_f.write(repr(vector).replace('\n', ' '))
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
                yield '%7.2f(%5.2f)' % (v, u)

        with open(path, 'w') as out_f:
            out_f.write('#Activations    Cluster Size   Act/Size   i   j \n')
            for activations, size, normalized_activations, center, vector, std in self.cluster_analysis():
                out_f.write('%10d    %10d    %8.2f    ' % (activations, size, normalized_activations))
                out_f.write('%3d %3d' % center)
                out_f.write('    ')
                # # De cyclify data if applicable
                # if cyclic is not None:
                #     # Need to update vector and std
                #     cyclic_vector = vector
                #     cyclic_std = std
                #     n_cyc = sum([int(b) for b in self.cyclic])
                #     noncyclic_vector = np.empty(len(cyclic_vector) - n_cyc)
                #     noncyclic_std = np.empty(len(cyclic_vector) - n_cyc)
                #     cyclic_feature_index = 0
                #     for feature_index, is_cyclic in enumerate(cyclic):
                #         if is_cyclic:
                #             x = cyclic_vector[cyclic_feature_index]
                #             y = cyclic_vector[cyclic_feature_index+1]
                #             v = self.decyclify(x, y,
                #                 cyclic_norms[feature_index], cyclic_modes[feature_index])
                #             cyclic_feature_index += 2
                #         else:
                #             noncyclic_vector[feature_index] = cyclic_vector[cyclic_feature_index]
                #             noncyclic_std[feature_index] = cyclic_std[cyclic_feature_index]
                #             cyclic_feature_index += 1
                #     vector = noncyclic_vector
                #     std = noncyclic_std
                for x in zip_error(vector, std):
                    out_f.write(x)
                    out_f.write(' ')
                out_f.write('\n')

    def find_center(self, cluster_indeces):
        index_array = np.array(cluster_indeces)
        mean_index = np.average(index_array, axis=0)
        diff = np.sum(np.abs(index_array - mean_index), axis = 1)
        closest_index = diff.argmin()
        return cluster_indeces[closest_index]

    def member_indeces(self, indeces, progress = False):
        """
        Returns a list of all the data IDs which are a member of this cluster.
        """
        members = []
        if progress:
            bar = progressbar.ProgressBar(max_value = len(self.data_n), redirect_stdout=True).start()
        for i, weight in enumerate(self.data_n):
            # Find out what the winner neuron is
            winner_index = self.winner(weight)
            if winner_index in indeces:
                members.append(i)
            if progress: bar.update(i)
        if progress: bar.finish()
        return members

    def member_IDs(self, indeces):
        members = self.member_indeces(indeces, progress = True)
        return [self.IDs[i] for i in members]

    def load_cluster_array(self, cluster_array):
        """
        Cluster arrays are booleans with value True where there is a
        cluster and False everywhere else.
        """
        assert cluster_array.shape == self.shape
        # Invert the cluster array so we can use threshold on it
        inverted = np.logical_not(cluster_array)
        # Find all the cluster indeces
        cluster_indeces = np.argwhere(cluster_array)
        # Cluster indeces should be a list of tuples
        cluster_indeces = [tuple(x) for x in cluster_indeces]
        # Now we need to break them up into different clusters
        clusters = []
        while len(cluster_indeces) > 0:
            # Pick an arbitrary cluster index
            center = cluster_indeces[0]
            # Find the cluster corresponding to it
            cluster_index_list = self.__cluster_indeces(inverted, center, 0.5)
            # Add these indeces to a cluster
            clusters.append(cluster_index_list)
            # Delete them from cluster_indeces
            cluster_indeces = [x for x in cluster_indeces if x not in cluster_index_list]
        # Now add every cluster
        for cluster_indeces in clusters:
            self.add_cluster(*self.cluster_weight(cluster_indeces[0], 0, \
                cluster_indeces = cluster_indeces))

    def selection_image_to_cluster_array(self, path):
        # Technically by the map coloring problem, we could use
        # the three color channels to have any configuration of
        # clusters we like, even allowing touching.
        image = plt.imread(path)[:,:,:3]
        # We need to find values that are black (those are in the cluster)
        # First collapse all the dimensions
        collapsed = np.sum(image, axis = 2)
        cluster_array = collapsed == 0
        return cluster_array

    def selection_image_to_cluster_array_triple(self, path):
        # Technically by the map coloring problem, we could use
        # the three color channels to have any configuration of
        # clusters we like, even allowing touching.
        image = plt.imread(path)[:,:,:3]
        # We need to find values that are saturated in a *single* color
        imager, imageg, imageb = (image == 255)
        not_more_than_one = + np.logical_not(imager + imageg) \
                            + np.logical_not(imager + imageb) \
                            + np.logical_not(imageg + imageb)
        imager = imager + not_more_than_one
        imageg = imageg + not_more_than_one
        imageb = imageb + not_more_than_one
        return imager, imageb, imageg

    def load_selection_image(self, path):
        self.load_cluster_array(self.selection_image_to_cluster_array(path))

    def __getitem__(self, indeces):
        return self._weights.__getitem__(indeces)

    def expand(self, n):
        """
            This returns an SOM weights matrix that is the same
            as this one but larger by a factor of n in both directions.
        """
        new_weights = np.repeat(np.repeat(self._weights, n, axis=0), n, axis=1)
        return new_weights