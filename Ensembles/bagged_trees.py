
import sys
import json
import numpy as np
import pandas as pd
import DecisionTree as dt

def bagged_trees(nTrees, maxDepth, train, test):
    """
    Bagging decision trees (ID3) to improve accuracy in classification

    Assumptions:
    - Solves classification problem
    - Decision Tree ID3 implementation provided by TAs of my Machine Learning course
    - Class (desired prediction) is last attribute in metadata and is named "class"

    Parameters
    ----------
    nTrees : number of trees to bag

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

    nTrees = int(nTrees)
    maxDepth = int(maxDepth)

    # Get training data
    train_meta = trainData['metadata']['features']
    train_data = np.array(trainData['data'])
    train_X = train_data[:,:-1]
    train_y = train_data[:,-1]
    copy_train_y = np.array(train_y)

    if isinstance(train_y[0],str):  # convert from string to int for coding purposes
        real_y = train_y
        uniqueClass = np.unique(train_y)
        for i in range(len(uniqueClass)):
            iIdx = [j for j, k in enumerate(real_y) if k == uniqueClass[i]]
            train_y[iIdx] = i


    # Get test data
    test_data = np.array(testData['data'])
    test_X = test_data[:,:-1]
    test_y = test_data[:,-1]
    copy_test_y = np.array(test_y)

    if isinstance(test_y[0], str):  # convert from string to int for coding purposes
        real_y = test_y
        uniqueClass = np.unique(test_y)
        for i in range(len(uniqueClass)):
            iIdx = [j for j, k in enumerate(real_y) if k == uniqueClass[i]]
            test_y[iIdx] = i


    ###################################################################
    ### Train and Test Model

    # could have pred output array have indices and then print the value?

    if train_meta[-1][1] != 'numeric':  # if class value categorical
        idxOut = np.zeros([train_X.shape[0],nTrees])  # initialize array for output of indices
        treePred = np.zeros([nTrees,test_X.shape[0],len(train_meta[len(train_meta) - 1][1])])
        predOut = pd.DataFrame(np.zeros([test_X.shape[0],nTrees + 2]))  # initialize array for output of predictions
        predOut.iloc[:, nTrees + 1] = copy_test_y  # last column of predicted output is actual test class values
        for t in range(nTrees):
            bootD = np.random.choice(train_X.shape[0], size=train_X.shape[0])  # sample indices w/ replacement
            idxOut[:,t] = bootD  # append output matrix
            tTree = dt.DecisionTree()  # initialize ID3 decision tree
            tTree.fit(train_X[bootD,:], copy_train_y[bootD], train_meta, max_depth=maxDepth)  # fit tree
            pred_y = tTree.predict(test_X, prob=True)  # predict classes
            treePred[t,:,:] = pred_y
            # predOut[:,t] = pred_y  # append prediction output
            classIdx = np.argmax(pred_y,axis=1)  # combine prediction
            for i in range(len(train_meta[-1][1])):  # iterate over class value indices
                iIdx = [j for j,k in enumerate(classIdx) if k == i]  # find indices which takes class at class index i
                predOut.iloc[iIdx,t] = train_meta[-1][1][i]  # class value for combined prediction

        avgProb = treePred.mean(axis=0)
        classIdx = np.argmax(avgProb, axis=1)  # combine prediction
        for i in range(len(train_meta[-1][1])):  # iterate over class value indices
            iIdx = [j for j, k in enumerate(classIdx) if k == i]  # find indices which takes class at class index i
            predOut.iloc[iIdx, nTrees] = train_meta[len(train_meta) - 1][1][i]  # class value for combined prediction

    acc = (predOut.iloc[:,nTrees]==predOut.iloc[:,nTrees+1]).mean()  # calculate test accuracy

    ###################################################################
    ### Print output

    for n in range(train_X.shape[0]):
        print(','.join(map(str, idxOut[n,:])))
    print("")

    for m in range(test_X.shape[0]):
        print(','.join(map(str, predOut.iloc[m,:])))
    print("")
    print(acc)



np.random.seed(0)
bagged_trees(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

