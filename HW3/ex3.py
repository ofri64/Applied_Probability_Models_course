# Ofri Kleinfeld    Shai Keynan 302893680   301687273

from DatasetHandler import DatasetHandler
from MixedHistogramMultinomialSmoothModel import MixedHistogramMultinomialSmoothModel
from EMAlgorithm import EMAlgorithm
# from Report import Report

dev_input = "dataset/develop.txt"
# dev_input = "develop.txt"
em = EMAlgorithm(num_clusters=9, k=10, lambda_=1, estimated_vocab_size=6800, stop_threshold=1)
em.run_algorithm(dev_input)

# At the end of algorithm write model parameters (theta) and iterations information
em.model.save_object_as_pickle()
DatasetHandler.write_results_to_file(em.iterations_likelihood, "iterations_likelihood.txt")
DatasetHandler.write_results_to_file(em.iterations_perplexity, "iterations_perplexity.txt")

# load train model from file and create report graphs and other assignments

trained_model = MixedHistogramMultinomialSmoothModel.load_model_object("model_object_lambda_0.01.pkl")
print(len(trained_model.frequent_words_set))
print(trained_model.k)
print(trained_model.lambda_)
#
# report = Report(trained_model, dev_input, "iterations_likelihood_lambda_0.01.txt", "iterations_perplexity_lambda_0.01.txt")
# report.plot_iterations_graphs("l")
# report.plot_iterations_graphs("p")
# report.create_confusion_mat()
#
# for i in range(trained_model.num_clusters):
#     report.plot_cluster_histogram(i)
#
# report.label_prediction_clusters()
# report.compute_model_accuracy()
