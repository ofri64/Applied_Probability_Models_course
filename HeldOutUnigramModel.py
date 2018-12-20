from AbstractUnigramModel import AbstractUnigramModel

# hels out unigram model
class HeldOutUnigramModel(AbstractUnigramModel):

    def __init__(self, estimated_vocab_size=300000):
        super(HeldOutUnigramModel, self).__init__()
        self.estimated_vocab_size = estimated_vocab_size
        self.frequency_classes = {}
        self.N0 = 0
        self.t0 = 0
        self.training_set_size = 0
        self.validation_set_size = 0

    # create and return a dict of {number of appearances: all words that appeared that  number of times in the set}
    def get_frequency_classes(self):

        if len(self.frequency_classes) == 0:

            # get training word counts
            training_word_counts = self.get_dataset_word_counts("train")

            # create frequency classes dictionary
            # key is frequency class (e.g 0, 1, 2, 3, ...)
            # frequency class 0 will get a special handle later on

            for word, frequency in training_word_counts.items():

                if frequency not in self.frequency_classes:
                    self.frequency_classes[frequency] = []

                self.frequency_classes[frequency].append(word)

        return self.frequency_classes

    # find the number of events that wasn't in the training set and were in the validation set
    def get_t0(self):

        # in this case we have to compute total number of times that events
        # unseen in training, appeared in validation
        if self.t0 <= 0:

            # get training word counts and number of unique tokens seen in training
            training_word_counts = self.get_dataset_word_counts("train")
            validation_word_counts = self.get_dataset_word_counts("validation")

            for word in validation_word_counts:
                if word not in training_word_counts:
                    self.t0 += validation_word_counts[word]

        # finally we return self.t0 (whether it was computed or just read from memory)
        return self.t0

    # calc N0- number of events that "unseen" in the data set
    def get_N0(self):

        # in this case we have to number of events unseen in training
        # we will counter our number of unique events and subtract it from our estimated vocabulary size
        if self.N0 <= 0:
            training_word_counts = self.get_dataset_word_counts("train")
            num_unique_words = len(training_word_counts.keys())
            self.N0 = self.estimated_vocab_size - num_unique_words

        # finally we return self.t0 (whether it was computed or just read from memory)
        return self.N0
    
    # calc Nr- number of different events that appeared r times in the set
    def get_Nr(self, r):

        if r == 0:
            return self.get_N0()

        else:
            frequency_classes = self.get_frequency_classes()
            return len(frequency_classes.get(r, []))

    # calc and return tr- number of appearances in validation set of all events that appeared r times in the training set
    def get_tr(self, r):

        if r == 0:
            return self.get_t0()

        else:
            tr = 0
            validation_word_counts = self.get_dataset_word_counts("validation")
            frequency_classes = self.get_frequency_classes()
            r_frequency_words = frequency_classes[r]

            for r_freq_word in r_frequency_words:
                tr += validation_word_counts.get(r_freq_word, 0)

            return tr

    # return number total number of tokens in the validation set
    def get_validation_set_size(self):

        if self.validation_set_size <= 0:
            self.validation_set_size = AbstractUnigramModel.get_num_events_for_dataset(self.validation_set_path)

        return self.validation_set_size

    # calc and return the propabilty of "word_token"s
    def get_token_prob(self, word_token):

        # first we compute "r" (frequency class) of given input word
        training_word_counts = self.get_dataset_word_counts("train")
        r = training_word_counts.get(word_token, 0)

        # resolve Nr, tr and |SH|
        Nr = self.get_Nr(r)
        tr = self.get_tr(r)
        validation_size = self.get_validation_set_size()

        # compute held out probability
        held_out_prob = tr / (validation_size * Nr)
        return held_out_prob

    # calc the fH- expected frequency according to the held_out prob
    def get_expected_frequency_for_frequency_class(self, r):

        # resolve Nr, tr and |SH|
        Nr = self.get_Nr(r)
        tr = self.get_tr(r)
        validation_size = self.get_validation_set_size()

        # compute held out probability
        held_out_prob = tr / (validation_size * Nr)

        # compute expected frequency of class r, according to the held out model
        training_word_counts = self.get_dataset_word_counts("train")
        expected_frequency = held_out_prob * sum(training_word_counts.values())

        return expected_frequency
