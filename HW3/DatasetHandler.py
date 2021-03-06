# Ofri Kleinfeld    Shai Keynan 302893680   301687273


class DatasetHandler(object):

    def __init__(self, input_path):

        self.input_path = input_path

    def generate_sentences(self):
        try:
            with open(self.input_path, "r") as fp:
                for line in fp:
                    line_tokens = line.split(" ")[:-1]
                    if not line_tokens:
                        continue
                    else:
                        yield line_tokens

        except Exception:
            print("Exception occurred, could not read file {0}".format(self.input_path))
            exit(-1)

    def generate_labeled_sentences(self):
        try:
            with open(self.input_path, "r") as fp:
                label = "NO_LABEL"
                for line in fp:
                    line_tokens = line.split(" ")[:-1]
                    if not line_tokens:  # now can be an empty line or header line with label

                        header_tokens = line[:-2].split("\t") # remove \n and '>' char
                        if len(header_tokens) > 1:
                            label = header_tokens[2:]

                        continue

                    else:
                        yield (line_tokens, label)

        except Exception:
            print("Exception occurred, could not read file {0}".format(self.input_path))
            exit(-1)

    def count_number_of_total_tokens(self, frequent_threshold=None):
        total_tokens = 0
        sentences_generator = self.generate_sentences()
        if not frequent_threshold:
            for sent in sentences_generator:
                total_tokens += len(sent)

        else:
            # have to perform two passes over the data - first to count frequencies and then sum only those above threshold
            raw_word_counts = {}
            for sent in sentences_generator:
                for word in sent:
                    raw_word_counts[word] = raw_word_counts.get(word, 0) + 1

            for word in raw_word_counts:
                if raw_word_counts[word] > frequent_threshold:
                    total_tokens += raw_word_counts[word]

        return total_tokens

    @staticmethod
    def write_results_to_file(results, output_file):
        with open(output_file, "w") as f:
            for res in results:
                f.write("{0}, ".format(res))
