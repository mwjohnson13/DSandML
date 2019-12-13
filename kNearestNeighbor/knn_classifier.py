import sys

def knn_classifier(k, train, test):
    """
    k-nearest neighbor classifier; using Hamming distance for discrete features and Manhattan distance for continuous features
    
    Assumptions:
    - Solves classification problem
    - Class (desired prediction) is last attribute in metadata and is named "label"
    
    Parameters
    ----------
    k : number of neighbors used to determine classification
    
    Data
    ----------
    train and test json files contain two parts, metadata and data
    metadata contains information in regards to feature name and type
    metadata is the same for train and test json files
    data are the train and test datasets
    
    Train data of 'm' instances and 'n' features
    
    Test data of 'M' instances and 'N' features
    """

    import json
    import numpy as np
    import pandas as pd

    k = int(k)

    # load train and test data
    with open(train) as data_file1:
        trainData = json.load(data_file1)

    with open(test) as data_file2:
        testData = json.load(data_file2)

    # determine if feature of data is numerical or categorical (excludes class label)
    # note: test data is same order as train
    fnames, ftypes = zip(*trainData['metadata']['features'])
    inum = [i for i, v in enumerate(ftypes[0:len(ftypes)-1]) if v == 'numeric']  # indices of numeric features
    icat = [i for i, v in enumerate(ftypes[0:len(ftypes)-1]) if v != 'numeric']  # indices of categorical features

    # convert data to numpy array
    trainD = np.array(trainData['data'])
    testD = np.array(testData['data'])

    xtrain = trainD[:, 0:trainD.shape[1] - 1]
    xtest = testD[:, 0:testD.shape[1] - 1]
    ytrain = trainD[:, trainD.shape[1]-1]

    # convert ytrain to integers if it is not already
    if isinstance(ftypes[len(ftypes)-1][0],str):
        realy = ytrain
        ytrain = np.zeros(len(realy))
        for i in range(len(np.unique(realy))):
            ytrain[np.where(realy==np.unique(realy)[i])] = i
        ytrain = ytrain.astype(int)

    # separate data set by feature type
    trainNum = xtrain[:, inum]
    testNum = xtest[:, inum]
    trainCat = xtrain[:, icat]
    testCat = xtest[:, icat]

    # standardize numeric data
    if trainNum.shape[1] != 0:
        trainstd = trainNum.std(axis=0, ddof=0)
        trainstd[trainstd == 0] = 1
        trainNorm = (trainNum - trainNum.mean(axis=0)) / trainstd
        testNorm = (testNum - trainNum.mean(axis=0)) / trainstd

    # testNum[:,None] transforms testNum from 2d (n x m) to 3d (n x 1 x m)
    # (same # rows but now one item of 1xm array in each)
    # trainNum - testNum[:,None] subtracts test rows from trainNum along columns
    # sum over features
    distCat = trainCat != testCat[:, None]  # hamming distance
    if trainNum.shape[1] != 0:
        distNum = abs(trainNorm - testNorm[:, None])  # manhattan distance
        distSumNum = distNum.sum(axis=2)
        distSumCat = distCat.sum(axis=2)
        dist = distSumNum + distSumCat  # (number test instances) x (number train instances)
    else:
        dist = distCat.sum(axis=2)  # (number test instances) x (number train instances)

    # sort features from smallest to largest and find k smallest distances
    # first k elements of distSort are the indices in dist of closest k instances
    distSort = np.argsort(dist, axis=1, kind='mergesort')
    knn = distSort[:, 0:k]  # indices of k-nearest neighbors; each row corresponds to test instance

    # k-nearest neighbor class values from training set
    choices = ytrain[knn]  # (#test instances) x k is the dimension

    # predicted class values and output
    numClass = len(trainData['metadata']['features'][len(trainData['metadata']['features'])-1][1])
    output = np.zeros([xtest.shape[0], numClass + 1]).astype(int)
    for j in range(choices.shape[0]):
        unique, counts = np.unique(choices[j, :], return_counts=True)
        output[j, unique] = counts
        output[j, numClass] = unique[np.argmax(counts)]

    # convert predicted class back to string if needed
    if isinstance(ftypes[len(ftypes)-1][0],str):
        predy = output[:, numClass].tolist()
        for i in range(len(predy)):
            for j in np.unique(output[:, numClass]):
                if predy[i] == j:
                    predy[i] = np.unique(realy)[j]
        output = output.astype(str)
        output[:, numClass] = predy

    output = output.tolist()
    for i in range(len(output)):
        print(','.join(str(x) for x in output[i]))

knn_classifier(sys.argv[1], sys.argv[2], sys.argv[3])





