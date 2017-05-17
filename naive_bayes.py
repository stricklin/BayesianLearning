#!/usr/bin/python3

import pandas as pd
import numpy as np
import math


def pdf(x, mean, std_dev):
    value = 1/(math.sqrt(2 * math.pi) * std_dev) * math.exp(- ((x - mean)**2) / (2 * std_dev ** 2))
    return value

# read data from files
training_data = pd.read_csv("./training_data.csv", header=None, index_col=57)
testing_data = pd.read_csv("./testing_data.csv", header=None, index_col=57)

num_features = len(training_data.columns)

# get the probability of mebership in a class
prob_pos = sum(training_data.index)/len(training_data)
prob_neg = 1 - prob_pos

training_data_targets = np.array(training_data.index.values)
training_data = np.array(training_data.values)
testing_data_targets = np.array(testing_data.index.values)
testing_data = np.array(testing_data.values)

# separate positive and negative instances
pos = []
neg = []
for i in range(len(training_data)):
    if training_data_targets[i] == 1:
        pos.append(training_data[i])
    else:
        neg.append(training_data[i])

pos = np.array(pos)
neg = np.array(neg)


# get the means and std dev for each feature and class combination
pos_means = pos.sum(0) / len(pos)
pos_std_devs = pos.std(0)
neg_means = neg.sum(0) / len(neg)
neg_std_devs = neg.std(0)

# correct for zeros
for i in range(len(pos_std_devs)):
    if pos_std_devs[i] == 0:
        pos_std_devs[i] = 0.0001
for i in range(len(neg_std_devs)):
    if neg_std_devs[i] == 0:
        neg_std_devs[i] = 0.0001

# classify test data
predictions = []
for i in range(len(testing_data)):
    is_pos = False
    is_neg = False
    # get the element to classify
    element = testing_data[i]
    # initalize the list of logs to sum
    pos_probs = [math.log(prob_pos)]
    neg_probs = [math.log(prob_neg)]
    # get the pdf of each feature in the element
    for k in range(len(element)):
        mean = pos_means[k]
        std_dev = pos_std_devs[k]
        value = pdf(element[k], pos_means[k], pos_std_devs[k])
        # if this elements feature made e^x = 0 with a very large negative x
        # classify it as negative
        if value == 0:
            is_neg = True
        else:
            pos_probs.append(math.log(pdf(element[k], pos_means[k], pos_std_devs[k])))
        value = pdf(element[k], neg_means[k], neg_std_devs[k])
        # if this elements feature made e^x = 0 with a very large negative x
        # classify it as poseitive
        if value == 0:
            is_pos = True
        else:
            neg_probs.append(math.log(pdf(element[k], neg_means[k], neg_std_devs[k])))
    # if not already classified by getting a negative infinity
    if not is_pos and not is_neg:
        pos_sum = sum(pos_probs)
        neg_sum = sum(neg_probs)
        if pos_sum > neg_sum:
            is_pos = True
        else:
            is_neg = True
    # collect classifications
    if is_pos:
        predictions.append(1)
    else:
        predictions.append(0)


# get confusion matrix:
total = len(testing_data_targets)
confusion_matrix = np.zeros((2, 2))
# columns are target class
# rows are predicted class
for i in range(total):
    confusion_matrix[predictions[i]][int(testing_data_targets[i])] += 1
# accuracy is true positives / total
accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / total
# precision is true positives / true positives + false positives for each class
pos_pres = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
neg_pres = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
# recall is true positives / true positives + false negatives
pos_recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
neg_recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

print("Confusion matrix:")
print(str(confusion_matrix))
print("Accuracy= " + str(accuracy))
print("Postive precision= " + str(pos_pres))
print("Negative precision= " + str(neg_pres))
print("Postive recall= " + str(pos_recall))
print("Negative recall= " + str(neg_recall))
