import math
from DatasetReader import DatasetReader


class MixedHistogramMultinomialSmoothModel(object):
    def __init__(self, num_clusters=9, lambda_=0.6, estimated_vocab_size=300000, epsilon_threshold = 0.001):
        self.num_clusters = num_clusters
        self.lambda_ = lambda_
        self.estimated_vocab_size = estimated_vocab_size
        self.epsilon_threshold = epsilon_threshold
        self.cluster_probs = []
        self.cluster_word_probs = [{} for i in range(num_clusters)]
        self.num_words_per_cluster = []

    def smooth_cluster_probs(self, current_cluster_probs):
        # change to epsilon if case of prob smaller than that
        for i in range(len(current_cluster_probs)):
            if current_cluster_probs[i] < self.epsilon_threshold:
                current_cluster_probs[i] = self.epsilon_threshold

        # we have to normalize again to a valid probability
        sum_of_current_probs = sum(current_cluster_probs)
        new_cluster_probs = [p / sum_of_current_probs for p in current_cluster_probs]

        return new_cluster_probs

    def initiate_word_and_cluster_probs(self, dataset_path, frequency_threshold=3):
        # init reader
        dataset_reader = DatasetReader(dataset_path)

        # init clusters and word in cluster counts data structures
        raw_word_counts = {}
        cluster_counts = [0 for i in range(self.num_clusters)]
        cluster_word_counts = [{} for i in range(self.num_clusters)]
        current_cluster = 0

        # iterate over dataset for first time
        # just count all words in dataset
        sent_generator = dataset_reader.generate_sentences()
        for sent in sent_generator:
            for word in sent:
                raw_word_counts[word] = raw_word_counts.get(word, 0) + 1

        # iterate over dataset for second time
        # now count the words for each topic separately
        # also use frequency threshold to reduce resources
        sent_generator = dataset_reader.generate_sentences()
        for sent in sent_generator:
            cluster_counts[current_cluster] += 1

            for word in sent:
                # apply frequency threshold reduction
                if raw_word_counts[word] > self.epsilon_threshold:

                    cluster_word_counts[current_cluster][word] = cluster_word_counts[current_cluster].get(word, 0) + 1

            # update current cluster
            current_cluster = (current_cluster + 1) % self.num_clusters

        # compute cluster probs from cluster counts
        # compute cluster word probs from cluster words counts
        num_sents = sum(cluster_counts)
        self.cluster_probs = [c / num_sents for c in cluster_counts]

        self.num_words_per_cluster = [sum(wc.values()) for wc in cluster_word_counts]
        for i in range(len(cluster_word_counts)):

            current_cluster_word_counts = cluster_word_counts[i]
            num_words_current_cluster = self.num_words_per_cluster[i]
            for word in current_cluster_word_counts:

                # we already smooth the data here using lidstone smoothing method
                self.cluster_word_probs[i][word] = \
                    (current_cluster_word_counts[word] + self.lambda_) / (num_words_current_cluster + self.lambda_ * self.estimated_vocab_size)

        # smooth the cluster probs
        self.cluster_probs = self.smooth_cluster_probs(self.cluster_probs)

    def get_p_xi(self, cluster_num):
        return self.cluster_probs[cluster_num]

    def get_p_w_given_xi(self, word, cluster_num):
        # the word was seen in training set
        # we return to smoothed probability
        if word in self.cluster_word_probs[cluster_num]:
            return self.cluster_word_probs[cluster_num][word]
        else:
            # return lidston probability for unseen words given topic i
            return self.lambda_ / (self.num_words_per_cluster[cluster_num] + self.lambda_ * self.estimated_vocab_size)

    def get_p_xi_given_sent(self, cluster_num, sentence):
        # initiate zi values according to numerically stability computation method
        k = 10
        z_values = [0 for i in range(self.num_clusters)]

        # sum log probabilities of each word in sentence according to each cluster
        for word in sentence:
            for i in range(self.num_clusters):
                z_values[i] += math.log(self.get_p_w_given_xi(word, i))

        # add alpha_i for each cluster
        for i in range(self.num_clusters):
            z_values[i] += math.log(self.get_p_xi(i))

        # use m and k to avoid unstable computations
        m = max(z_values)
        z_minus_m_values = [z-m for z in z_values]
        e_values = [math.exp(z - m) for z in z_values]

        if z_minus_m_values[cluster_num] < -k:
            # value is too small and may cause underflow
            return 0

        else:
            normalization_denominator = 0
            for i in range(len(z_values)):
                # add to sum only numerical stable, else assume they are zero
                if z_minus_m_values[i] >= -k:
                    normalization_denominator += e_values[i]

            return e_values[cluster_num] / normalization_denominator
