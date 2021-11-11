import math
import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Node(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def __str__(self, level=0):
        ret = "\t|" * level + "+----" + repr(self.name) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


def print_nested_dict(nested_dict):
    for attr in nested_dict:
        print(attr)
        for i in nested_dict[attr]:
            print("\t ", i, "\t[", nested_dict[attr][i][0], "+ ,", nested_dict[attr][i][1], "- ]")


def discretization(x):
    col = []
    # for each row of that specific column
    for i in range(x.shape[0]):
        col.append(x[i, 0])
    # sort the column array so the first index contains min value, last index contains max value for that column
    col.sort()
    min_of_col = col[0]
    max_of_col = col[x.shape[0] - 1]

    interval = (max_of_col - min_of_col + 1) / 5

    # optimize here
    for j in range(x.shape[0]):
        # 16->30
        if min_of_col <= x[j, 0] < min_of_col + interval:
            x[j, 0] = "group1"

        # 30->45
        elif min_of_col + interval <= x[j, 0] < min_of_col + (2 * interval):
            x[j, 0] = "group2"

        # 45->60
        elif min_of_col + (2 * interval) <= x[j, 0] < min_of_col + (3 * interval):
            x[j, 0] = "group3"

        # 60->75
        elif min_of_col + (3 * interval) <= x[j, 0] < min_of_col + (4 * interval):
            x[j, 0] = "group4"

        # 75->90
        elif min_of_col + (4 * interval) <= x[j, 0] < min_of_col + (5 * interval):
            x[j, 0] = "group5"
    return x


# nested_dict = { 'dictA': {'key_1': 'value_1'},
#                 'dictB': {'key_2': 'value_2'}}
def find_class_distribution(x, attributes):
    class_distribution = {}

    for attr in attributes:
        col = attributes.get(attr)

        if col == 0:
            age_dict = {"group1": [0, 0],
                        "group2": [0, 0],
                        "group3": [0, 0],
                        "group4": [0, 0],
                        "group5": [0, 0],
                        "total": [0, 0]
                        }

            # for each row of that specific column
            for row in range(len(x)):
                if x[row, col] == "group1":
                    if x[row, 16] == "Positive":
                        age_dict["group1"][0] += 1
                    else:
                        age_dict["group1"][1] += 1

                elif x[row, col] == "group2":
                    if x[row, 16] == "Positive":
                        age_dict["group2"][0] += 1
                    else:
                        age_dict["group2"][1] += 1

                elif x[row, col] == "group3":
                    if x[row, 16] == "Positive":
                        age_dict["group3"][0] += 1
                    else:
                        age_dict["group3"][1] += 1

                elif x[row, col] == "group4":
                    if x[row, 16] == "Positive":
                        age_dict["group4"][0] += 1
                    else:
                        age_dict["group4"][1] += 1

                elif x[row, col] == "group5":
                    if x[row, 16] == "Positive":
                        age_dict["group5"][0] += 1
                    else:
                        age_dict["group5"][1] += 1

            age_dict["total"][0] += age_dict["group1"][0]
            age_dict["total"][0] += age_dict["group2"][0]
            age_dict["total"][0] += age_dict["group3"][0]
            age_dict["total"][0] += age_dict["group4"][0]
            age_dict["total"][0] += age_dict["group5"][0]

            age_dict["total"][1] += age_dict["group1"][1]
            age_dict["total"][1] += age_dict["group2"][1]
            age_dict["total"][1] += age_dict["group3"][1]
            age_dict["total"][1] += age_dict["group4"][1]
            age_dict["total"][1] += age_dict["group5"][1]

            class_distribution["age"] = age_dict

        elif col == 1:
            gender_dict = {"male": [0, 0],
                           "female": [0, 0],
                           "total": [0, 0]}

            # for each row of that specific column
            for row in range(len(x)):
                if x[row, col] == "Male":
                    if x[row, 16] == "Positive":
                        gender_dict["male"][0] += 1
                    else:
                        gender_dict["male"][1] += 1

                elif x[row, col] == "Female":
                    if x[row, 16] == "Positive":
                        gender_dict["female"][0] += 1
                    else:
                        gender_dict["female"][1] += 1

            gender_dict["total"][0] += gender_dict["male"][0]
            gender_dict["total"][0] += gender_dict["female"][0]

            gender_dict["total"][1] += gender_dict["male"][1]
            gender_dict["total"][1] += gender_dict["female"][1]

            class_distribution["gender"] = gender_dict

        else:
            attr_dict = {"yes": [0, 0],
                         "no": [0, 0],
                         "total": [0, 0]}

            # for each row of that specific column
            for row in range(len(x)):
                if x[row, col] == "Yes":
                    if x[row, 16] == "Positive":
                        attr_dict["yes"][0] += 1
                    else:
                        attr_dict["yes"][1] += 1

                elif x[row, col] == "No":
                    if x[row, 16] == "Positive":
                        attr_dict["no"][0] += 1
                    else:
                        attr_dict["no"][1] += 1

            attr_dict["total"][0] += attr_dict["yes"][0]
            attr_dict["total"][0] += attr_dict["no"][0]

            attr_dict["total"][1] += attr_dict["yes"][1]
            attr_dict["total"][1] += attr_dict["no"][1]

            class_distribution[attr] = attr_dict

    # print_nested_dict(class_distribution)
    return class_distribution


# dist (distribution) is like positive,negative count [ 13+ , 23- ]
def calculate_entropy(dist):
    total = dist[0] + dist[1]
    if total == 0:
        return 0
    pos_prop = dist[0] / total
    neg_prop = dist[1] / total
    log = lambda prop: math.log(prop, 2) if prop != 0 else 0
    entropy = -(pos_prop * log(pos_prop)) - (neg_prop * log(neg_prop))
    return entropy


def calculate_info_gain(dist, attributes):
    info_gain = {"age": 0, "gender": 0, "polyuria": 0, "polydipsia": 0, "sudden weight loss": 0,
                 "weakness": 0, "polyphagia": 0, "penital thrush": 0, "visual blurring": 0,
                 "itching": 0, "irritability": 0, "delayed healing": 0, "partial paresis": 0,
                 "muscle stiffness": 0, "alopecia": 0, "obesity": 0}
    for attr in attributes:
        gain = calculate_entropy(dist.get(attr).get("total"))
        for i in dist.get(attr):
            if i != "total":
                sv = dist.get(attr).get(i)[0] + dist.get(attr).get(i)[1]
                s = dist.get(attr).get("total")[0] + dist.get(attr).get("total")[1]
                ent_sv = calculate_entropy(dist.get(attr).get(i))
                gain -= abs(sv / s) * ent_sv
        info_gain[attr] = gain
    return info_gain


def select_next_node(info_gain):
    max_info_gain = 0
    best_attr = ""
    for x in info_gain:
        if info_gain.get(x) > max_info_gain:
            best_attr = x
            max_info_gain = info_gain.get(x)
    return best_attr


def find_most_frequent(data):
    positive_count = 0
    negative_count = 0
    for i in range(len(data)):
        if data[i, 16] == "Positive":
            positive_count += 1
        else:
            negative_count += 1
    return positive_count, negative_count


def ID3(data, rem_features):
    positive, negative = find_most_frequent(data)
    if positive > negative:
        guess = "Positive"
    else:
        guess = "Negative"

    if positive == 0:
        return Node("LEAF-Negative", ["Negative"])
    elif negative == 0:
        return Node("LEAF-Positive", ["Positive"])
    elif len(rem_features) == 0:
        return Node("LEAF-" + guess, [guess])
    else:
        class_distribution = find_class_distribution(data, rem_features)
        info_gain = calculate_info_gain(class_distribution, rem_features)
        node_name = select_next_node(info_gain)
        if node_name == "":
            return Node("LEAF-" + guess, [guess])
        # print("---------------------------->",node_name)
        node_values = []
        # # print(".....................",class_distribution)
        for x in class_distribution.get(node_name):
            if x != "total":
                node_values.append(x)

        features = rem_features.copy()
        rem_features.pop(node_name)
        # print(rem_features.keys())
        node_children = []

        for i in node_values:  # yes no
            subset = numpy.ndarray([0, 17])
            for row in range(len(data)):
                val = data[row, features.get(node_name)]
                if val.lower() == i:
                    subset = np.vstack([subset, data[row, :]])

            node_children.append(ID3(subset, rem_features.copy()))

        node = Node(node_name, node_values)
        node.children = node_children
        return node


def decision_tree_test(node, test):
    attributes = {"age": 0, "gender": 1, "polyuria": 2, "polydipsia": 3, "sudden weight loss": 4,
                  "weakness": 5, "polyphagia": 6, "penital thrush": 7, "visual blurring": 8,
                  "itching": 9, "irritability": 10, "delayed healing": 11, "partial paresis": 12,
                  "muscle stiffness": 13, "alopecia": 14, "obesity": 15}

    if node.name == "LEAF-Positive" or node.name == "LEAF-Negative":
        return node.values[0]
    else:
        for val in node.values:
            if test[attributes.get(node.name)].lower() == val:
                index = node.values.index(val)
                child_node = node.children[index]
                return decision_tree_test(child_node, test)


def classification_performance(root, x_test):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for sample in x_test:
        prediction = decision_tree_test(root, sample)
        if prediction == sample[16] and prediction == "Positive":
            tp += 1
        elif prediction == sample[16] and prediction == "Negative":
            tn += 1
        if prediction != sample[16] and prediction == "Positive":
            fp += 1
        elif prediction != sample[16] and prediction == "Negative":
            fn += 1
    print("TP = ", tp)
    print("TN = ", tn)
    print("FP = ", fp)
    print("FN = ", fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * recall * precision) / (recall + precision)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)


def k_fold(x):
    # start and end points of each fold
    size = int(x.shape[0] / 5)
    arr = [0, size, 2 * size, 3 * size, 4 * size, 5 * size]
    # for each fold, we create our test and train set and then call KNN classification function
    for i in range(5):
        # 1/5 part of the data set as test data
        x_test = x[arr[i]:arr[i + 1]]

        # rest of the data set as train data
        a = x[0:arr[i]]
        b = x[arr[i + 1]:]
        x_train = np.concatenate((a, b), axis=0)

        attributes = {"age": 0, "gender": 1, "polyuria": 2, "polydipsia": 3, "sudden weight loss": 4,
                      "weakness": 5, "polyphagia": 6, "penital thrush": 7, "visual blurring": 8,
                      "itching": 9, "irritability": 10, "delayed healing": 11, "partial paresis": 12,
                      "muscle stiffness": 13, "alopecia": 14, "obesity": 15}

        print()
        print("--------------------------FOLD", i + 1, "--------------------------------------------")

        root = ID3(x_train, attributes)
        print(root)

        classification_performance(root, x_test)

    return


def main():
    # reading data's in the csv file to the numpy array
    df = pd.read_csv('./diabetes_data_upload.csv')
    x = np.array(df.iloc[:, :])

    # discretization on age attribute
    x = discretization(x)

    # shuffle the data
    np.random.seed(101)
    np.random.shuffle(x)

    k_fold(x.copy())


if __name__ == "__main__":
    main()
