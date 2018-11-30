import math
from AbstractUnigramModel import AbstractUnigramModel


class LidstoneUnigramModel(AbstractUnigramModel):

    def __init__(self, lambda_=0, estimated_vocab_size=300000):

        super(LidstoneUnigramModel, self).__init__()
        self.lambda_ = lambda_
        self.estimated_vocab_size = estimated_vocab_size
        self.total_training_tokens = 0

    def get_token_prob(self, word_token):

        if len(self.training_word_counts) == 0:
            raise AssertionError("Must count number of token in training set before performing inference")

        if self.total_training_tokens == 0:
            self.total_training_tokens = sum(self.training_word_counts.values())

        count_word = self.training_word_counts.get(word_token, 0)
        extended_sample_size = self.total_training_tokens + self.lambda_ * self.estimated_vocab_size
        return (count_word + self.lambda_) / extended_sample_size

    def get_MLE_estimator(self, word_token):

        current_lambda = self.lambda_

        # for MLE "Vanila" estimator, lambda equal zero
        self.lambda_ = 0
        estimator = self.get_token_prob(word_token)

        # don't forget to return lambda to it's original value
        self.lambda_ = current_lambda

        return estimator

    def set_lambda(self, new_lambda):

        self.lambda_ = new_lambda

    def grid_search_lambda(self, lambda_values):

        try:
            current_lambda = self.lambda_
            best_lambda = lambda_values[0]
            min_perplexity_score = math.inf

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
