from EMAlgorithm import EMAlgorithm
dev_input = "dataset/develop.txt"

em = EMAlgorithm(num_clusters=9)
em.run_algorithm(dev_input)
