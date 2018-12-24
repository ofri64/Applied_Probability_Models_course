import math
from DatasetReader import DatasetReader
from MixedHistogramMultinomialSmoothModel import MixedHistogramMultinomialSmoothModel


class EMAlgorithm(object):
    def __init__(self, num_clusters=9, stop_threshold=10):
        self.num_clusters = num_clusters
        self.stop_threshold = stop_threshold
        self.model = MixedHistogramMultinomialSmoothModel(num_clusters=self.num_clusters)

    def run_algorithm(self, training_set_path):
        # initiate the values for theta - model parameters
        self.model.initiate_word_and_cluster_probs(training_set_path)

        # initiate data reader and other variables
        data_reader = DatasetReader(training_set_path)
        num_total_training_tokens = data_reader.count_number_of_total_tokens()
        iteration_num = 1
        prev_likelihood = 0
        current_likelihood = self._compute_likelihood(training_set_path)

        print("likelihood value for iteration {0} is: {1}".format(iteration_num, current_likelihood))

        # iterate until stopping criterion
        while abs(current_likelihood - prev_likelihood) > self.stop_threshold:

            # E part is already implemented within the model class
            # perform M part to update model parameters

            # update cluster probs
            new_cluster_probs = [0 for i in range(self.num_clusters)]
            sentences = data_reader.generate_sentences()

            for sent in sentences:
                for i in range(self.num_clusters):
                    new_cluster_probs[i] += self.model.get_p_xi_given_sent(i, sent)

            # divide by N and normalize/smooth
            new_cluster_probs = [count / num_total_training_tokens for count in new_cluster_probs]
            new_cluster_probs = self.model.smooth_cluster_probs(new_cluster_probs)

            # update word cluster probs




            # assign new cluster probs to model
            # self.model.cluster_probs = new_cluster_probs

    def _compute_likelihood(self, training_set_path):
        log_likelihood = 0
        data_reader = DatasetReader(training_set_path)
        sentences_generator = data_reader.generate_sentences()
        k = 10

        for sent in sentences_generator:
            z_values = [0 for i in range(self.num_clusters)]
            for word in sent:
                for i in range(self.num_clusters):
                    z_values[i] += math.log(self.model.get_p_w_given_xi(word, i))

            # add alpha_i for each cluster
            for i in range(self.num_clusters):
                z_values[i] += math.log(self.model.get_p_xi(i))

            m = max(z_values)
            z_minus_m_values = [z - m for z in z_values]

            # add log of only numerically stable components
            sent_stable_comp_sum = 0
            for z_m in z_minus_m_values:
                if z_m >= -k:
                    sent_stable_comp_sum += math.exp(z_m)

            log_likelihood += m + math.log(sent_stable_comp_sum)

        return log_likelihood
