from DatasetHandler import DatasetHandler
from EMAlgorithm import EMAlgorithm

dev_input = "dataset/develop.txt"
em = EMAlgorithm(num_clusters=9)
em.run_algorithm(dev_input)

# At the end of algorithm write model parameters (theta) and iterations information
em.model.save_object_as_pickle()
DatasetHandler.write_results_to_file(em.iterations_likelihood, "iterations_likelihood.txt")
DatasetHandler.write_results_to_file(em.iterations_perplexity, "iterations_perplexity.txt")
