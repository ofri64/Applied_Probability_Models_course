import math
from DatasetHandler import DatasetHandler
from MixedHistogramMultinomialSmoothModel import MixedHistogramMultinomialSmoothModel


class EMAlgorithm(object):
    def __init__(self, num_clusters=9, stop_threshold=0.000001, max_num_iterations=300):
        self.num_clusters = num_clusters
        self.stop_threshold = stop_threshold
        self.max_num_iterations = max_num_iterations
        self.model = MixedHistogramMultinomialSmoothModel(num_clusters=self.num_clusters)
        self.iterations_likelihood = []
        self.iterations_perplexity = []

    def run_algorithm(self, training_set_path):
        # initiate the values for theta - model parameters
        self.model.initiate_word_and_cluster_probs(training_set_path)

        # initiate data reader and other variables
        data_reader = DatasetHandler(training_set_path)
        num_total_training_tokens = data_reader.count_number_of_total_tokens()
        iteration_num = 0
        prev_likelihood = -99999999
        current_likelihood = self._compute_likelihood(training_set_path)
        current_perplexity = self._compute_perplexity(current_likelihood, num_total_training_tokens)
        self.iterations_likelihood = [current_likelihood]
        self.iterations_perplexity = [current_perplexity]

        print("likelihood value for iteration {0} is: {1}".format(iteration_num, current_likelihood))
        print("perplexity value for iteration {0} is: {1}".format(iteration_num, current_perplexity))

        # iterate until stopping criterion
        while (current_likelihood - prev_likelihood) / abs(prev_likelihood) > self.stop_threshold and iteration_num < self.max_num_iterations:
            iteration_num += 1
            # print("starting iteration {0}".format(iteration_num))

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

            # print("finished updating cluster probs")

            # update word cluster probs

            # initiate needed helper variables
            cluster_word_mass = [{} for i in range(self.num_clusters)]
            sentences = data_reader.generate_sentences()

            # now we sum all the "mass" for every word that is frequent enough, given the current model theta
            for sent in sentences:
                for i in range(self.num_clusters):
                    current_model_p_xi_given_sent = self.model.get_p_xi_given_sent(i, sent)
                    for word in sent:
                        if word in self.model.frequent_words_set:  # if word is frequent enough
                            cluster_word_mass[i][word] = cluster_word_mass[i].get(word, 0) + current_model_p_xi_given_sent

            clusters_total_mass = [sum(cluster_word_mass[i].values()) for i in range(self.num_clusters)]

            # now compute probs using the mass
            # apply also lidston smoothing over the calculated probabilites
            cluster_word_probs = [{} for i in range(self.num_clusters)]
            lambda_ = self.model.lambda_
            vocab_size = self.model.estimated_vocab_size

            for word in self.model.frequent_words_set:
                for i in range(self.num_clusters):
                    current_cluster_word_mass = cluster_word_mass[i][word]
                    current_cluster_total_mass = clusters_total_mass[i]
                    cluster_word_probs[i][word] = (current_cluster_word_mass + lambda_) / (current_cluster_total_mass + vocab_size * lambda_)

            # print("finished updating cluster word probs")

            # assign new cluster probs to and new word cluster probs to model

            self.model.cluster_probs = new_cluster_probs
            self.model.cluster_word_probs = cluster_word_probs
            self.model.em_clusters_total_mass = clusters_total_mass

            # print and save current likelihood and new likelihood

            prev_likelihood = current_likelihood
            current_likelihood = self._compute_likelihood(training_set_path)
            current_perplexity = self._compute_perplexity(current_likelihood, num_total_training_tokens)
            self.iterations_likelihood.append(current_likelihood)
            self.iterations_perplexity.append(current_perplexity)
            print("likelihood value for iteration {0} is: {1}".format(iteration_num, current_likelihood))
            print("perplexity value for iteration {0} is: {1}".format(iteration_num, current_perplexity))

        # At the end of algorithm write model parameters (theta) and iterations information
        self.model.save_object_as_pickle()
        DatasetHandler.write_results_to_file(self.iterations_likelihood, "iterations_likelihood.txt")

    def _compute_likelihood(self, training_set_path):
        log_likelihood = 0
        data_reader = DatasetHandler(training_set_path)
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

    def _compute_perplexity(self, log_likelihood, total_training_tokens):
        average_inverse_likelihood = -1 * log_likelihood / total_training_tokens
        return math.exp(average_inverse_likelihood)
