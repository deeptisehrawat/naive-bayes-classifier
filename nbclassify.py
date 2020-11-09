import sys
import json
import os


def classify_data(classify_data_path):
    model = open('nbmodel.txt', 'r')
    params_dict = json.load(model)
    model.close()

    output_fp = open('nboutput.txt', 'w')
    for dirpath, dirnames, filenames in os.walk(classify_data_path, topdown=True):
        for name in filenames:
            if name.endswith(".txt") and "readme" not in name.lower():
                pos_true = params_dict["PosTruePriorProb"]
                pos_dec = params_dict["PosDeceptivePriorProb"]
                neg_true = params_dict["NegTruePriorProb"]
                neg_dec = params_dict["NegDeceptivePriorProb"]

                words = open(os.path.join(dirpath, name), 'r').read().split(" ")
                for idx in range(len(words)):
                    curr_word = words[idx].lower()
                    pos_true += params_dict["PosTrueWordProb"].get(curr_word, 0)
                    pos_dec += params_dict["PosDeceptiveWordProb"].get(curr_word, 0)
                    neg_true += params_dict["NegTrueWordProb"].get(curr_word, 0)
                    neg_dec += params_dict["NegDeceptiveWordProb"].get(curr_word, 0)

                label_1, label_2 = "deceptive", "negative"
                max_prob = max([pos_true, pos_dec, neg_true, neg_dec])
                if max_prob == neg_true:
                    label_1, label_2 = "truthful", "negative"
                elif max_prob == pos_dec:
                    label_1, label_2 = "deceptive", "positive"
                elif max_prob == pos_true:
                    label_1, label_2 = "truthful", "positive"

                output_fp.write("{} {} {}\n".format(label_1, label_2, os.path.join(dirpath, name)))

    output_fp.close()


if __name__ == "__main__":
    classify_data(sys.argv[1])
