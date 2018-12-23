from HW2.AbstractUnigramModel import AbstractUnigramModel

# lidstone unigram model
class LidstoneUnigramModel(AbstractUnigramModel):

    def __init__(self, lambda_=0, estimated_vocab_size=300000):

        super(LidstoneUnigramModel, self).__init__()
        self.lambda_ = lambda_
        self.estimated_vocab_size = estimated_vocab_size
        self.total_training_tokens = -1

    # return number total number of tokens in the training set
    def get_total_training_tokens(self):

        if self.total_training_tokens < 0:
            train_word_counts = self.get_dataset_word_counts("train")
            self.total_training_tokens = sum(train_word_counts.values())

        return self.total_training_tokens

    # calc and return the propabilty of "word_token" using "lambda" in the training set
    def get_token_prob(self, word_token):

        total_training_tokens = self.get_total_training_tokens()

        count_word = self.training_word_counts.get(word_token, 0)
        extended_sample_size = total_training_tokens + self.lambda_ * self.estimated_vocab_size
        return (count_word + self.lambda_) / extended_sample_size

    # calc and return the MLE estimator by the training set with get_token_prob method as lamnda=0
    def get_MLE_estimator(self, word_token):

        current_lambda = self.lambda_

        # for MLE "Vanila" estimator, lambda equal zero
        self.lambda_ = 0
        estimator = self.get_token_prob(word_token)

        # don't forget to return lambda to it's original value
        self.lambda_ = current_lambda

        return estimator

    # set lambda
    def set_lambda(self, new_lambda):

        self.lambda_ = new_lambda

    # find the best lambda that gives the min perplexity by calc perplexity with each lambda value in the given set
    def grid_search_lambda(self, lambda_values):

        try:
            current_lambda = self.lambda_
            best_lambda = lambda_values[0]
            min_perplexity_score = 99999

            for lambda_ in lambda_values:
                self.set_lambda(lambda_)
                perplexity_score = self.calculate_perplexity(self.validation_set_path)

                # update if score is best up until now
                if perplexity_score < min_perplexity_score:
                    best_lambda = lambda_
                    min_perplexity_score = perplexity_score

            self.set_lambda(current_lambda)
            return best_lambda, min_perplexity_score

        except (TypeError, IndexError) as err:
            print("Error in lambda values range: {0}".format(err))
            return None, None

        except FileNotFoundError as fnf:
            print("You must perform train-validation split before performing grid search")
            print("The error was: {0}".format(fnf.strerror))
            return None, None

    def get_expected_frequency_for_frequency_class(self, r):

        total_training_tokens = self.get_total_training_tokens()
        extended_sample_size = total_training_tokens + self.lambda_ * self.estimated_vocab_size
        frequency_class_prob = (r + self.lambda_) / extended_sample_size
        expected_frequency = frequency_class_prob * total_training_tokens

        return expected_frequency

