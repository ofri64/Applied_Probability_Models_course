import sys
from AbstractUnigramModel import AbstractUnigramModel
from LidstonUnigramModel import LidstoneUnigramModel
from HeldOutUnigramModel import HeldOutUnigramModel

VOCAB_SIZE = 300000


class OutputWriter(object):

    def __init__(self, output_file_path):
        self.output_file_path = output_file_path
        self.current_line_counter = 1
        self.fp = open(self.output_file_path, "w", encoding="utf8")

    def write_line(self, value):
        self.fp.write("#Output{0}:\t{1}\n".format(self.current_line_counter, value))
        self.current_line_counter += 1

    def write_students(self, students_names, students_ids):
        self.fp.write("#Students\t{0}\t{1}\t{2}\t{3}\n".format(*students_names, *students_ids))

    def close_file(self):
        self.fp.close()


if __name__ == '__main__':

    num_args_supplied = len(sys.argv) - 1
    if num_args_supplied != 4:
        usage = "The program must receive as input 4 arguments - development file, test file, input word and output file name"
        print(usage)
        exit(-1)

    dev_file, test_file, input_word, output_file = sys.argv[1:]
    students_names = ["Ofri Kleinfeld", "Ofri Kleinfeld"]
    students_ids = ["302893680", "302893680"]

    output_writer = OutputWriter(output_file)

    # write input arguments details
    output_writer.write_students(students_names, students_ids)
    output_writer.write_line(dev_file)
    output_writer.write_line(test_file)
    output_writer.write_line(input_word)
    output_writer.write_line(output_file)
    output_writer.write_line(VOCAB_SIZE)

    uniform_prob = 1 / VOCAB_SIZE
    output_writer.write_line(uniform_prob)

    dev_set_events = AbstractUnigramModel.get_num_events_for_dataset(dev_file)
    output_writer.write_line(dev_set_events)

    # lidston model part
    lidstone = LidstoneUnigramModel()
    training_validation_ratio = 0.9
    num_train, num_validation = lidstone.split_dev_to_train_validation(dev_file, training_validation_ratio)
    output_writer.write_line(num_validation)
    output_writer.write_line(num_train)

    train_words_count = lidstone.get_dataset_word_counts("train")
    num_unique_tokens_train = len(train_words_count.keys())
    output_writer.write_line(num_unique_tokens_train)

    input_word_freq = train_words_count.get(input_word, 0)
    output_writer.write_line(input_word_freq)

    # MLE estimator
    unseen_word = "unseen-word"
    input_word_estimator = lidstone.get_MLE_estimator(input_word)
    unseen_word_estimator = lidstone.get_MLE_estimator(unseen_word)
    output_writer.write_line(input_word_estimator)
    output_writer.write_line(unseen_word_estimator)

    # estimators using different lambda values
    lidstone.set_lambda(0.1)
    input_word_estimator = lidstone.get_token_prob(input_word)
    unseen_word_estimator = lidstone.get_token_prob(unseen_word)
    output_writer.write_line(input_word_estimator)
    output_writer.write_line(unseen_word_estimator)

    # perplexity score of different lambda
    lambda_values = [0.01, 0.1, 1]
    for lambda_ in lambda_values:
        lidstone.set_lambda(lambda_)
        lambda_perplexity = lidstone.calculate_perplexity(lidstone.validation_set_path)
        output_writer.write_line(lambda_perplexity)

    # grid search best lambda on validation set
    grid_search_lambda_values = [x * 0.01 for x in range(200)][1:]  # avoid zero lambda
    best_lambda, min_perplexity = lidstone.grid_search_lambda(grid_search_lambda_values)
    output_writer.write_line(best_lambda)
    output_writer.write_line(min_perplexity)

    # Held-Out model part

    held_out = HeldOutUnigramModel()
    training_validation_ratio = 0.5
    num_train, num_validation = held_out.split_dev_to_train_validation(dev_file, training_validation_ratio)
    held_out.validation_set_size = num_validation

    # write train and held out set sizes
    output_writer.write_line(num_train)
    output_writer.write_line(num_validation)

    # write estimators for input word and unseen word according to held out smoothing
    input_word_estimator = held_out.get_token_prob(input_word)
    unseen_word_estimator = held_out.get_token_prob(unseen_word)
    output_writer.write_line(input_word_estimator)
    output_writer.write_line(unseen_word_estimator)

    # models evaluation and comparison on test file

    num_events_test_set = AbstractUnigramModel.get_num_events_for_dataset(test_file)
    lidstone.set_lambda(best_lambda)
    best_lidstone_test_perplexity = lidstone.calculate_perplexity(test_file)
    held_out_test_perplexity = held_out.calculate_perplexity(test_file)

    output_writer.write_line(best_lidstone_test_perplexity)
    output_writer.write_line(held_out_test_perplexity)

    better_model = 'H' if held_out_test_perplexity < best_lidstone_test_perplexity else 'L'
    output_writer.write_line(better_model)

    output_writer.close_file()



