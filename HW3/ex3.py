from EMAlgorithm import EMAlgorithm
# from MixedHistogramMultinomialSmoothModel import MixedHistogramMultinomialSmoothModel
# from DatasetReader import DatasetReader
dev_input = "dataset/develop.txt"
# dataset_reader = DatasetReader(dev_input)

em = EMAlgorithm(num_clusters=9)
em.run_algorithm(dev_input)

# model = MixedHistogramMultinomialSmoothModel(num_clusters=9)
# model.initiate_word_and_cluster_probs(dev_input)
# sentences_generator = dataset_reader.generate_sentences()
# sent1 = next(sentences_generator)
# sent2 = next(sentences_generator)
# sent3 = next(sentences_generator)
#
# print(model.get_p_xi_given_sent(0, sent1))
# print(model.get_p_xi_given_sent(4, sent1))
# sanity_sam = 0
# for i in range(9):
#     sanity_sam += model.get_p_xi_given_sent(i, sent3)
#     print(i, model.get_p_xi_given_sent(i, sent3))
#
# print(sanity_sam)


# datasetReader = DatasetReader(dev_input)
# counter = 0
# raw_generator = datasetReader.generate_sentences()
# labeled_generator = datasetReader.generate_labeled_sentences()
# while counter < 10:
#     print(next(raw_generator))
#     print(next(labeled_generator)[-1])
#     counter += 1
