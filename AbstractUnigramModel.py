import os
import math


class AbstractUnigramModel(object):

    def __init__(self):

        self.training_set_path = None
        self.validation_set_path = None
        self.training_word_counts = {}
        self.validation_word_counts = {}

    @staticmethod
    def get_data_lines_generator(dataset_path):

        try:
            with open(dataset_path, "r") as fp:
                for line in fp:
                    line_tokens = line.split(" ")[:-1]
                    if not line_tokens:
                        continue
                    else:
                        yield line_tokens

        except Exception:
            print("Exception occurred, could not read file {0}".format(dataset_path))
            exit(-1)

    @staticmethod
    def get_num_events_for_dataset(dataset_path):

        num_events = 0
        lines_generator = AbstractUnigramModel.get_data_lines_generator(dataset_path)

        for line_tokens in lines_generator:
            num_events += len(line_tokens)

        return num_events

    def split_dev_to_train_validation(self, validation_set_path, train_ratio):

        # calculate how many tokens should be in training set and validation set
        num_events = AbstractUnigramModel.get_num_events_for_dataset(validation_set_path)
        num_train = round(train_ratio * num_events)
        num_validation = num_events - num_train

        # initiate file paths for to-be-written training and validation files
        self.training_set_path = os.path.join(os.curdir, "training_set.txt")
        self.validation_set_path = os.path.join(os.curdir, "validation_set.txt")

        # write line to training set file - up until we reach number of needed tokens
        current_num_training = 0
        write_to_training = True

        training_fp = open(self.training_set_path, "w", encoding="utf8")
        validation_fp = open(self.validation_set_path, "w", encoding="utf8")

        lines_generator = AbstractUnigramModel.get_data_lines_generator(validation_set_path)

        for line_tokens in lines_generator:
            if write_to_training:
                if current_num_training + len(line_tokens) < num_train:
                    training_fp.write(" ".join(line_tokens + ["\n"]))
                    current_num_training += len(line_tokens)
                else:
                    # finish writing to training set
                    num_write_to_training = num_train - current_num_training
                    training_fp.write(" ".join(line_tokens[:num_write_to_training] + ["\n"]))
                    current_num_training += num_write_to_training

                    # mark writing to train is done and write leftover of line to validation
                    write_to_training = False
                    validation_fp.write(" ".join(line_tokens[num_write_to_training:] + ["\n"]))

            else:
                # write to validation entire line
                validation_fp.write(" ".join(line_tokens + ["\n"]))

        training_fp.close()
        validation_fp.close()

        # return number of training tokens and number of validation tokens
        return num_train, num_validation

    def get_dataset_word_counts(self, dataset_type):
        if dataset_type not in ["train", "validation"]:
            raise AttributeError("Data set type of calculating word counts must be either train or test")

        if (dataset_type == "train" and self.training_set_path is None) or (dataset_type == "validation" and self.validation_set_path is None):
            raise FileNotFoundError("Cannot find path for data set of type {0}".format(dataset_type))

        file_path, word_counts = (self.training_set_path, self.training_word_counts) \
            if dataset_type == "train" else (self.validation_set_path, self.validation_word_counts)

        if len(word_counts) != 0:
            # already performed data set tokens enumeration
            return word_counts

        # count frequency of tokens in dataset
        lines_generator = AbstractUnigramModel.get_data_lines_generator(file_path)

        for line_tokens in lines_generator:
            for token in line_tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        return word_counts

    def get_token_prob(self, word_token):
        raise NotImplementedError("Each language model must implement get token prob method")

    def calculate_perplexity(self, dataset_path):
        num_total_tokens = 0
        sum_of_log_probs = 0

        lines_generator = AbstractUnigramModel.get_data_lines_generator(dataset_path)
        for line_tokens in lines_generator:

            for token in line_tokens:

                # add log of current token prob to log sum
                if token == "afterwards":
                    a = 1
                sum_of_log_probs += math.log(self.get_token_prob(token), 2)
                # update number of token counts
                num_total_tokens += 1

        perplexity = math.pow(2, (-1/num_total_tokens) * sum_of_log_probs)
        return perplexity

