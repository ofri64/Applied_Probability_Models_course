# Ofri Kleinfeld    Shai Keynan 302893680   301687273

from DatasetHandler import DatasetHandler
from MixedHistogramMultinomialSmoothModel import MixedHistogramMultinomialSmoothModel
from EMAlgorithm import EMAlgorithm

dev_input = "dataset/develop.txt"
em = EMAlgorithm(num_clusters=9)
em.run_algorithm(dev_input)

# At the end of algorithm write model parameters (theta) and iterations information
# em.model.save_object_as_pickle()
# DatasetHandler.write_results_to_file(em.iterations_likelihood, "iterations_likelihood.txt")
# DatasetHandler.write_results_to_file(em.iterations_perplexity, "iterations_perplexity.txt")

# trained_model = MixedHistogramMultinomialSmoothModel.load_model_object()
# print(len(trained_model.frequent_words_set))