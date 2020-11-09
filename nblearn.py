import json
import math
import sys
from glob import glob

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "he", "him", "his", "himself", "she",
              "her", "hers", "herself", "you", "your", "yours", "yourself", "yourselves", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "have", "has", "had",
              "having", "do", "does", "did", "doing", "what", "which", "who", "whom", "this", "that",
              "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "a", "an", "the", "and",
              "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "then", "once",
              "here", "there", "when", "where", "why", "how",
              "all", "about", "against", "between", "into", "through", "only", "own", "same", "so", "than", "too",
              "very", "s", "t", "can", "will", "just", "don", "should", "now",
              "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
              "over", "under", "again", "further", "any", "both", "each", "few", "more", "most", "other", "some",
              "such", "no", "nor", "not", "didnt", "dont", "doesnt", "isnt", "arent", "wasnt", "werent", "havent",
              "hasnt", "hadnt", "shouldnt"
              }


# extract words from files and create word dictionaries to store frequency of each word
def preprocess_files(fp, dictionary, stop_words):
    files = []
    files += glob(fp)
    word_count_dict = {}

    # get all files from file path
    for file_name in files:
        for file in glob(file_name + "*.txt"):
            text = open(file, 'r').read()
            text = text.lower()  # change all text to lowercase
            for word in text.split(" "):
                if word not in stop_words:  # if word is not in stop word, add to dictionary of unique words, add count
                    word_count_dict[word] = word_count_dict.get(word, 0) + 1
                    dictionary.add(word)
    return word_count_dict, len(files)


# calculate probabilities of each word in class (positiveTrue, positiveDeceptive, negativeTrue, negativeDeceptive)
def calculate_prob(word_count_dict, dictionary, dictionary_len, label_dict_len):
    for word in dictionary:
        count = word_count_dict.get(word, 0)
        word_count_dict[word] = math.log2(float((count + 1) / (label_dict_len + dictionary_len)))
    return word_count_dict


# read data, learn the model and write learned parameters to ndmodel.txt file
def learn_model(train_dir):
    pos_true = train_dir + "/positive_polarity/truthful_from_TripAdvisor/*/"
    pos_deceptive = train_dir + "/positive_polarity/deceptive_from_MTurk/*/"
    neg_true = train_dir + "/negative_polarity/truthful_from_Web/*/"
    neg_deceptive = train_dir + "/negative_polarity/deceptive_from_MTurk/*/"

    dictionary = set()
    pt_word_count_dict, pt_doc_count = preprocess_files(pos_true, dictionary, stop_words)
    pd_word_count_dict, pd_doc_count = preprocess_files(pos_deceptive, dictionary, stop_words)
    nt_word_count_dict, nt_doc_count = preprocess_files(neg_true, dictionary, stop_words)
    nd_word_count_dict, nd_doc_count = preprocess_files(neg_deceptive, dictionary, stop_words)

    dictionary_len = len(dictionary)
    total_doc_count = nd_doc_count + nt_doc_count + pd_doc_count + pt_doc_count

    pt_word_count_dict = calculate_prob(pt_word_count_dict, dictionary, dictionary_len, len(pt_word_count_dict))
    pd_word_count_dict = calculate_prob(pd_word_count_dict, dictionary, dictionary_len, len(pd_word_count_dict))
    nt_word_count_dict = calculate_prob(nt_word_count_dict, dictionary, dictionary_len, len(nt_word_count_dict))
    nd_word_count_dict = calculate_prob(nd_word_count_dict, dictionary, dictionary_len, len(nd_word_count_dict))

    # store parameters in a map
    parameters = {
        'PosTruePriorProb': math.log2(pt_doc_count / total_doc_count),
        'PosDeceptivePriorProb': math.log2(pd_doc_count / total_doc_count),
        'NegTruePriorProb': math.log2(nt_doc_count / total_doc_count),
        'NegDeceptivePriorProb': math.log2(nd_doc_count / total_doc_count),
        'PosTrueWordProb': pt_word_count_dict,
        'PosDeceptiveWordProb': pd_word_count_dict,
        'NegTrueWordProb': nt_word_count_dict,
        'NegDeceptiveWordProb': nd_word_count_dict
    }

    # write parameters to the model file
    model_fp = open("nbmodel.txt", 'w')
    model_fp.write(json.dumps(parameters))
    model_fp.close()


if __name__ == "__main__":
    learn_model(sys.argv[1])
    # learn_model("./op_spam_training_data")
