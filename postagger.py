from collections import defaultdict
import numpy as np
from optparse import OptionParser
import logging, re


def is_NNS(word):
    is_nns = False

    if word[-2:] == 'es' or word[-2:] == 'ES' or (word[:-1].isupper() and word[-1] == 's'):
        is_nns = True
    if word[-1] == 's' or word[-1] == 'S':
        if len(word) > 4:
            if 'ness' != word[-4:] or 'NESS' != word[-4:]:
                is_nns = False
        else:
            is_nns = True

    return is_nns


def is_adjective(word):
    is_ad = False

    word = word.lower()

    if 4 < len(word):

        if word[-1] == 's':
            if word[-4:] != 'ness' and (
                    word[-4:] == 'less' or word[-4:] == 'eous' or word[-4:] == 'ious' or word[-3:] == 'ous'):
                is_ad = True

        elif ('y' == word[-1] or word[-2:] == 'ly') or '-' in word or word[-4:] == 'able' \
                or word[-4:] == 'ible' or word[-2:] == 'al' or word[-3:] == 'ial' or word[-3:] == 'ful' \
                or word[-2:] == 'ic' or word[-4:] == 'ical' or word[-3:] == 'ish' \
                or word[-3:] == 'ive' or word[-5:] == 'ative' or word[-3:] == 'ian':
            is_ad = True
        elif word[:2] == 'un' and (word[-3:] == 'ing' or word[-2:] == 'ed'):
            is_ad = True

    return is_ad


def morphology(word):
    if re.match(number_pattern, word):
        expected_tag = "CD"

    elif 'ance' == word[-4:] or 'ence' == word[-4:]:
        expected_tag = 'NN'

    elif is_adjective(word):
        expected_tag = 'JJ'

    elif (word[0].isupper() and word[1:].islower() and word[-2:] != 'ed' and word[-3:] != 'ing' and word[-1] != 's') \
            or word.isupper():
        expected_tag = 'NNP'

    elif (word[0].isupper() and word[1:].islower() and word[-2:] != 'ed' and word[-3:] != 'ing' and word[
        -1] == 's'):
        expected_tag = 'NNS'

    elif word[:2] != 'un' and word[-2:] == 'ed' and '-' not in word:
        expected_tag = 'VBD'

    elif word[-3:] == 'ing' and word[:2] != 'un':
        expected_tag = 'VBG'

    else:
        if is_NNS(word):
            return 'NNS'
        else:
            expected_tag = 'NN'

    return expected_tag


def acc(seq1, seq2):
    acc = 0

    for i in range(len(seq1)):
        if seq1[i][1] == seq2[i][1]:
            acc += 1

    return acc / len(seq1)


def calc_prob(dic, count, smooth=False):
    for key in dic.keys():
        for k in dic[key].keys():
            if smooth:
                dic[key][k] = (dic[key][k] + 1) / (count[key] + 46)
            else:
                dic[key][k] /= count[key]


number_pattern = r'^[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?$'


def read_tokens(file):
    f = open(file)
    sentences = []
    for l in f.readlines():
        tokens = l.split()
        sentence = []
        for token in tokens:
            try:
                word, tag = token.rsplit('/', 1)
            except:
                word, tag = token, "UNK"
            sentence.append((word, tag))
        sentences.append(sentence)
    return sentences


def viterbi(sentence, tag_tag, tag_word, word_count, tag_set, pi):
    viterbi_mat = np.zeros((len(list(tag_set)), len(sentence), 2))
    T = len(tag_set)
    # col 0
    word = sentence[0][0]

    if word_count[word] == 0:

        expected_tag = morphology(word)
        index = tag_set.index(expected_tag)
        viterbi_mat[index][0][0] = pi[expected_tag]
    else:
        for i in range(len(list(tag_set))):
            viterbi_mat[i][0][0] = pi[tag_set[i]] * tag_word[tag_set[i]][word]

    # ----------------------------------------------------------------

    for j in range(1, len(sentence)):
        word = sentence[j][0]

        if word_count[word] == 0:

            expected_tag = morphology(word)
            index = tag_set.index(expected_tag)

            max_val = 0
            argmax = 0

            for k in range(T):

                v, t = viterbi_mat[k][j - 1][0], tag_tag[tag_set[k]][tag_set[index]]

                if v > 0 and t > 0:

                    val = v * t

                    if val > max_val:
                        max_val = val
                        argmax = k

            viterbi_mat[index][j][0] = max_val
            viterbi_mat[index][j][1] = argmax

        else:

            for i in range(T):

                max_val = 0
                argmax = 0
                if tag_word[tag_set[i]][word] > 0:
                    for k in range(T):
                        if viterbi_mat[k][j - 1][0] > 0 and tag_tag[tag_set[k]][tag_set[i]] > 0:
                            val = viterbi_mat[k][j - 1][0] * tag_tag[tag_set[k]][tag_set[i]] * tag_word[tag_set[i]][
                                word]

                            if val > max_val:
                                max_val = val
                                argmax = k

                viterbi_mat[i][j][0] = max_val
                viterbi_mat[i][j][1] = argmax

    # backtrack

    S = len(sentence)
    row = np.argmax(viterbi_mat[:, S - 1, 0])
    col = S - 1
    tokens = []
    for i in range(S):
        token = [sentence[col], tag_set[int(row)]]
        tokens.append(token)
        row = viterbi_mat[int(row)][col][1]
        col -= 1

    return tokens[::-1]


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

    sentences = read_tokens(training_file)
    test_sentences = read_tokens(test_file)

    tag_tag = defaultdict(lambda: defaultdict(int))
    tag_word = defaultdict(lambda: defaultdict(int))
    tag_set = defaultdict(int)
    vocabulary = defaultdict(int)

    pi = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            token = sentence[i]
            word = token[0]
            tag = token[1]
            tag_set[tag] += 1
            vocabulary[word] += 1

            if i == 0:
                pi[tag] += 1

            tag_word[tag][word] += 1
            tag_tag[tag][sentence[i + 1][1]] += 1

    for tag in pi.keys():
        pi[tag] /= tag_set[tag]

    calc_prob(tag_tag, tag_set)
    calc_prob(tag_word, tag_set)
    test_accuracy = 0
    train_accuracy = 0

    for sentence in test_sentences:
        tag_seq = viterbi(sentence, tag_tag, tag_word, vocabulary, list(tag_set.keys()), pi)
        test_accuracy += acc(sentence, tag_seq)

    for sentence in sentences:
        tag_seq = viterbi(sentence, tag_tag, tag_word, vocabulary, list(tag_set.keys()), pi)

        train_accuracy += acc(sentence, tag_seq)

    print('Accuracy in training [' + str(len(sentences)) + ' sentences]:',
          str((train_accuracy / len(sentences)) * 100) + '%')

    print('Accuracy in test [' + str(len(test_sentences)) + ' sentences]:',
          str((test_accuracy / len(test_sentences)) * 100) + '%')
