import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


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
def find_class_distribution(x):
    class_distribution = {}
    # for 16 attributes (0 to 16)
    for col in range(x.shape[1] - 1):

        if col == 0:
            age_dict = {"group1": [0, 0],
                        "group2": [0, 0],
                        "group3": [0, 0],
                        "group4": [0, 0],
                        "group5": [0, 0],
                        "total": [0, 0]
                        }

            # for each row of that specific column
            for row in range(x.shape[0]):
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
            for row in range(x.shape[0]):
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
            for row in range(x.shape[0]):
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

            attr_name = ["age", "gender", "polyuria", "polydipsia", "sudden weight loss", "weakness", "polyphagia",
                         "penital thrush", "visual blurring", "itching", "irritability", "delayed healing",
                         "partial paresis", "muscle stiffness", "alopecia", "obesity"]

            attr_dict["total"][0] += attr_dict["yes"][0]
            attr_dict["total"][0] += attr_dict["no"][0]

            attr_dict["total"][1] += attr_dict["yes"][1]
            attr_dict["total"][1] += attr_dict["no"][1]

            class_distribution[attr_name[col]] = attr_dict

    print_nested_dict(class_distribution)
    return class_distribution


# dist is 4,- count [ 13+ , 23- ]
def calculate_entropy(dist):
    total = dist[0] + dist[1]
    pos_prop = dist[0] / total
    neg_prop = dist[1] / total
    entropy = -(pos_prop * math.log(pos_prop, 2)) - (neg_prop * math.log(neg_prop, 2))
    print(entropy)
    return entropy


def calculate_info_gain(proportions):
    return


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

        print()
        print("--------------------------FOLD", i + 1, "--------------------------------------------")
        print(x_train.shape)
        print(x_test.shape)
        class_distribution = find_class_distribution(x_train)
        calculate_entropy(class_distribution.get("age").get("group1"))

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

    # df = pd.DataFrame(x)
    # df.to_csv('file_name.csv')

    k_fold(x.copy())


if __name__ == "__main__":
    main()
