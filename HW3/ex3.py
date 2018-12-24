from MixedHistogramMultinomialSmoothModel import MixedHistogramMultinomialSmoothModel
dev_input = "dataset/develop.txt"

model = MixedHistogramMultinomialSmoothModel(num_clusters=9)
model.initiate_word_and_cluster_probs(dev_input)
print(model.cluster_probs)
print(sum(model.cluster_probs))



# datasetReader = DatasetReader(dev_input)
# counter = 0
# raw_generator = datasetReader.generate_sentences()
# labeled_generator = datasetReader.generate_labeled_sentences()
# while counter < 10:
#     print(next(raw_generator))
#     print(next(labeled_generator)[-1])
#     counter += 1
