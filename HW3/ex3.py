from DatasetReader import DatasetReader

dev_input = "dataset/develop.txt"
datasetReader = DatasetReader(dev_input)

counter = 0
raw_generator = datasetReader.generate_sentences()
labeled_generator = datasetReader.generate_labeled_sentences()
while counter < 10:
    print(next(raw_generator))
    print(next(labeled_generator)[-1])
    counter += 1
