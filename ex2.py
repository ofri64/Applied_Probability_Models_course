import sys
from AbstractUnigramModel import AbstractUnigramModel
from LidstonUnigramModel import LidstoneUnigramModel

if __name__ == '__main__':

    # num_args_supplied = len(sys.argv) - 1
    # if num_args_supplied != 4:
    #     usage = "The program must receive as input 4 arguments - development file, test file, input word and output file name"
    #     print(usage)
    #     exit(-1)

    dev_file = "/Users/okleinfeld/PycharmProjects/Applied_Probability_Models_course/data_files/dataset/develop.txt"
    print(AbstractUnigramModel.get_num_events_for_dataset(dev_file))
    lidstone = LidstoneUnigramModel()
    num_train, num_test = lidstone.split_dev_to_train_validation(dev_file)

    train_words_count = lidstone.get_dataset_word_counts("train")
    validation_words_count = lidstone.get_dataset_word_counts("validation")

    print(AbstractUnigramModel.get_num_events_for_dataset("./training_set.txt"))
    print(AbstractUnigramModel.get_num_events_for_dataset("./validation_set.txt"))
    num_train_events = sum(train_words_count.values())
    num_validation_events = sum(validation_words_count.values())
    num_unique_tokens_train = len(train_words_count.keys())

    print(num_train_events)
    print(num_validation_events)
    print(num_unique_tokens_train)

