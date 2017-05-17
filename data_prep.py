#!/usr/bin/python3
import pandas as pd
import numpy as np
import random

if __name__ == "__main__":
    # read from file
    data = pd.read_csv("./spambase/spambase.data", header=None)

    # turn everything into np.arrays because DataFrames are weird
    data = np.array(data)

    # divide positive and negative examples
    pos = []
    neg = []
    for i in range(len(data)):
        if data[i][57] == 1:
            pos.append(data[i])
        else:
            neg.append(data[i])

    # shuffle data
    random.shuffle(pos)
    random.shuffle(neg)

    # divide both in half then combine pos and neg
    training = pos[:len(pos)//2] + neg[:len(neg)//2]
    testing = pos[len(pos)//2:] + neg[len(neg)//2:]

    # shuffle data
    random.shuffle(training)
    random.shuffle(testing)

    # write data to file
    training = np.array(training)
    testing = np.array(testing)
    np.savetxt("training_data.csv", training, delimiter=',')
    np.savetxt("testing_data.csv", testing, delimiter=',')


