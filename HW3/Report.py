import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DatasetHandler import DatasetHandler


class Report(object):
    def __init__(self, trained_model, test_path, likelihood_file, perplexity_file):
        self.model = trained_model
        self.data_handler = DatasetHandler(test_path)
        self.likelihood_file = likelihood_file
        self.perplexity_file = perplexity_file
        self.clusters_topics = []
        self.ordered_topics = ["acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
        self.clusters_predicted_labels = []

    @staticmethod
    def load_list_from_file(input_file):
        with open(input_file, "r") as f:
            values_as_string = f.readline()
            list_of_strings = values_as_string.split(",")[:-1]
            return [float(v) for v in list_of_strings]

    def plot_iterations_graphs(self, type):
        if type == "p":
            data_type = "Perplexity"
            data = self.load_list_from_file(self.perplexity_file)
            color = 'red'
        else:
            data_type = "Log Likelihood"
            data = self.load_list_from_file(self.likelihood_file)
            color = 'blue'

        title = '{0} as a function of iteration number'.format(data_type)
        iterations = range(1, len(data)+1)
        plt.style.use('bmh')
        plt.plot(iterations, data, color=color, alpha=0.4)
        plt.xlabel('Iteration number')
        plt.ylabel(data_type)
        plt.title(title)
        plt.grid(True)
        plt.savefig("{0}.png".format(data_type))
        plt.close()

    def create_confusion_mat(self):
        clusters_topics = [{} for i in range(self.model.num_clusters)]
        labeled_sents_generator = self.data_handler.generate_labeled_sentences()

        for sent, labels in labeled_sents_generator:
            predicted_cluster = self.model.classify_sent(sent)
            for label in labels:
                clusters_topics[predicted_cluster][label] = clusters_topics[predicted_cluster].get(label, 0) + 1

        df = pd.DataFrame(clusters_topics)
        df = df[self.ordered_topics]
        df = df.fillna(0)
        df["total_in_cluster"] = df.sum(axis=1)
        self.clusters_topics = clusters_topics
        df.to_csv("conf_matrix_lambda_0.01.csv")
        print(df)

    def label_prediction_clusters(self):
        if len(self.clusters_topics) == 0:
            self.create_confusion_mat()

        for cluster_topics_dict in self.clusters_topics:
            sorted_cluster_topics = sorted(cluster_topics_dict.items(), key=lambda x: x[1], reverse=True)
            self.clusters_predicted_labels.append(sorted_cluster_topics[0])

        print(self.clusters_predicted_labels)

    def plot_cluster_histogram(self, cluster_num):
        title = 'Cluster number {0} topics histogram'.format(cluster_num)
        topics = self.ordered_topics
        counts = [self.clusters_topics[cluster_num].get(topic, 0) for topic in topics]
        plt.style.use('bmh')
        plt.bar(topics, counts, color='green', alpha=0.3)
        plt.xlabel('Topics')
        plt.title(title)
        plt.grid(True)
        plt.savefig("cluster_{0}_histogram.png".format(cluster_num))
        plt.close()

    def compute_model_accuracy(self):
        if len(self.clusters_predicted_labels) == 0:
            self.label_prediction_clusters()

        num_correct_predicted = 0
        total_sents = 0
        labeled_sents_generator = self.data_handler.generate_labeled_sentences()

        for sent, labels in labeled_sents_generator:
            predicted_cluster = self.model.classify_sent(sent)
            cluster_label = self.clusters_predicted_labels[predicted_cluster][0]

            if cluster_label in labels:
                num_correct_predicted += 1

            total_sents += 1

        accuracy = num_correct_predicted / total_sents
        print("The accuracy of the model is {0}".format(accuracy))
        return accuracy
