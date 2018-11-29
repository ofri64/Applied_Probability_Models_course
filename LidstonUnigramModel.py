from AbstractUnigramModel import AbstractUnigramModel


class LidstoneUnigramModel(AbstractUnigramModel):

    ESTIMATED_VOCAB_SIZE = 300000

    def __init__(self, lambda_=0):

        super(LidstoneUnigramModel, self).__init__()
        self.lambda_ = lambda_
        self.total_training_tokens = 0

    def get_token_prob(self, word_token):

        if len(self.training_word_counts) == 0:
            raise AssertionError("Must count number of token in training set before performing inference")

        if self.total_training_tokens == 0:
            self.total_training_tokens = sum(self.training_word_counts.values())

        count_word = self.training_word_counts.get(word_token, 0)
        extended_sample_size = self.total_training_tokens + self.lambda_ * LidstoneUnigramModel.ESTIMATED_VOCAB_SIZE
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
