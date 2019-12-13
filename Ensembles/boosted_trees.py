
import sys
import json
import numpy as np
import pandas as pd
import DecisionTree as dt
import math

def boosted_trees(maxTrees, maxDepth, train, test):
    """
    Boosting decision trees (ID3 and AdaBoost) to improve accuracy in classification

    Assumptions:
    - Solves classification problem
    - Decision Tree ID3 implementation provided by TAs of my Machine Learning course
    - Class (desired prediction) is last attribute in metadata and is named "class"

    Parameters
    ----------
    maxTrees : number of trees in ensemble

    maxDepth : maximum depth of each ID3 tree

    Data
    ----------
    train and test json files contain two parts, metadata and data
        metadata contains information in regards to feature name and type
        metadata is the same for train and test json files
        data are the train and test datasets

    Train data of 'm' instances and 'n' features

    Test data of 'M' instances and 'N' features
    """

    with open(train) as data_file1:
        trainData = json.load(data_file1)

    with open(test) as data_file2:
        testData = json.load(data_file2)

    ###################################################################
    ### Data Pre-processing

    maxTrees = int(maxTrees)
    maxDepth = int(maxDepth)

    # Get training data
    train_meta = trainData['metadata']['features']
    train_data = np.array(trainData['data'])
    train_X = train_data[:,:-1]
    train_y = train_data[:,-1]
    copy_train_y = np.array(train_y)

    if isinstance(train_y[0],str):  # convert from string to int for coding purposes
       uniqueClass = np.unique(train_y)
       real_y = train_y
       for i in range(len(uniqueClass)):
           iIdx = [j for j, k in enumerate(real_y) if k == uniqueClass[i]]
           train_y[iIdx] = i


    # Get test data
    test_data = np.array(testData['data'])
    test_X = test_data[:,:-1]
    test_y = test_data[:,-1]
    copy_test_y = np.array(test_y)

    if isinstance(test_y[0], str):  # convert from string to int for coding purposes
        uniqueClass = np.unique(test_y)
        real_y = test_y
        for i in range(len(uniqueClass)):
            iIdx = [j for j, k in enumerate(real_y) if k == uniqueClass[i]]
            test_y[iIdx] = i

    K = len(train_meta[len(train_meta)-1][1])  # number of classes

    ###################################################################
    ### Train and Test Model

    T = np.array(range(maxTrees))
    trainWeight = np.zeros([train_X.shape[0], maxTrees])  # initialize array for train weight output
    treeWeight = np.zeros(maxTrees)  # initialize array for tree weights output
    # treePred = np.zeros([maxTrees, test_X.shape[0], len(train_meta[len(train_meta) - 1][1])])
    treePred = pd.DataFrame(np.zeros([test_X.shape[0], maxTrees]))
    predOut = pd.DataFrame(np.zeros([test_X.shape[0], maxTrees + 2]))  # initialize array for output of predictions
    predOut.iloc[:, maxTrees + 1] = copy_test_y  # last column of predicted output is actual test class values
    w = np.ones(train_X.shape[0]) / train_X.shape[0]  # initialize weights
    for t in T:
        trainWeight[:,t] = w
        tTree = dt.DecisionTree()
        tTree.fit(train_X, copy_train_y, train_meta, max_depth=maxDepth, instance_weights=w)  # fit tree
        err = (w * (tTree.predict(train_X) != copy_train_y) / w.sum()).sum()
        if err >= 1 - 1/K:
            T = t-1  # T = t - 1?
            break
        alpha = math.log((1-err)/err) + math.log(K-1)
        w = w * np.exp(alpha * (tTree.predict(train_X) != copy_train_y))  # update weights
        w = w / w.sum()  # renormalize weights
        treeWeight[t] = alpha
        pred_y = tTree.predict(test_X)  # predict classes
        treePred.iloc[:, t] = pred_y
        predOut.iloc[:,t] = pred_y

    # C = np.zeros(test_X.shape[0])  # initialize output vector
    argK = np.zeros([test_X.shape[0],K])
    # for i in range(test_X.shape[0]):
    #     Ck = np.zeros(K)  # initialize possible class choices
    #     for k in range(K):
    #         Ck[k] = treeWeight * (tTree.predict(train_X)[i] == k)
    for k in range(K):
        if isinstance(test_y[0], str):
            argK[:, k] = (treeWeight * (treePred == train_meta[-1][1][k])).sum(axis=1)
        else:
            argK[:,k] = (treeWeight * (treePred == int(train_meta[-1][1][k]))).sum(axis=1)  # summed weight for argument k

    # predOut[:, maxTrees] = np.argmax(argK,axis=1)  # need to sort out indexing and class value when they're not equal
    # need to sort out indexing and class value when they're not equal
    classIdx = np.argmax(argK,axis=1)
    for i in range(K):  # iterate over class value indices
        iIdx = [j for j, k in enumerate(classIdx) if k == i]  # find indices which takes class at class index i
        predOut.iloc[iIdx, maxTrees] = train_meta[-1][1][i]  # class value for combined prediction

    acc = (predOut.iloc[:, maxTrees] == predOut.iloc[:, maxTrees + 1]).mean()  # calculate test accuracy

    ###################################################################
    ### Print output

    for n in range(train_X.shape[0]):
        print(','.join(map(str, trainWeight[n, :])))
    print("")

    print(','.join(map(str,treeWeight)))
    print("")

    for m in range(test_X.shape[0]):
        print(','.join(map(str, predOut.iloc[m, :])))
    print("")
    print(acc)



np.random.seed(0)
boosted_trees(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

