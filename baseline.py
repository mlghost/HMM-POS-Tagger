from collections import defaultdict
from optparse import OptionParser
import logging


def read_tokens(file):
    f = open(file)
    sentences = []
    for l in f.readlines():
        tokens = l.split()
        sentence = []
        for token in tokens:
            ## split only one time, e.g. pianist|bassoonist\/composer/NN
            try:
                word, tag = token.rsplit('/', 1)
            except:
                ## no tag info (test), assign tag UNK
                word, tag = token, "UNK"
            sentence.append((word, tag))
        sentences.append(sentence)
    return sentences


if __name__ == "__main__":
    usage = "usage: %prog [options] GOLD TEST"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--debug", action="store_true",
                      help="turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Please provide required arguments")

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    training_file = args[0]
    test_file = args[1]

    tags = defaultdict(lambda: defaultdict(int))
    words = defaultdict(lambda: defaultdict(int))

    tokens = read_tokens(training_file)
    for sentence in tokens:
        for token in sentence:
            word = token[0]
            tag = token[1]
            tags[tag][word] += 1
            words[word][tag] += 1

    test_tokens = read_tokens(test_file)
    seq = []
    acc_test = 0
    acc_train = 0
    c = 0
    c1 = 0
    for sentence in test_tokens:
        predicted_seq = []
        for token in sentence:
            word = token[0]
            tag = token[1]
            if word not in words.keys():
                predicted_tag = 'NN'
            else:
                predicted_tag = sorted(list(words[word].items()), key=lambda x: x[1], reverse=True)[0][0]

            if predicted_tag == tag:
                acc_test += 1
            c += 1

    for sentence in tokens:
        predicted_seq = []
        for token in sentence:
            word = token[0]
            tag = token[1]
            if word not in words.keys():
                predicted_tag = 'NN'
            else:
                predicted_tag = sorted(list(words[word].items()), key=lambda x: x[1], reverse=True)[0][0]

            if predicted_tag == tag:
                acc_train += 1
            c1 += 1

    print('Accuracy in training [' + str(len(tokens)) + ' sentences]:', str((acc_train / c1) * 100) + '%')
    print('Accuracy in test [' + str(len(test_tokens)) + ' sentences]:', str((acc_test / c) * 100) + '%')
