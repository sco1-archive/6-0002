# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:45:20 2016

@author: johnguttag
"""

import numpy as np
import pylab
import random
import sklearn
from sklearn.linear_model import LogisticRegression

pylab.rcParams['lines.linewidth'] = 4       # width of lines in plots
pylab.rcParams['axes.titlesize'] = 20       # font size for titles
pylab.rcParams['axes.labelsize'] = 20       # font size for labels on axes
pylab.rcParams['xtick.labelsize'] = 16      # size of numbers on x-axis
pylab.rcParams['ytick.labelsize'] = 16      # size of numbers on y-axis
pylab.rcParams['xtick.major.size'] = 7      # size of ticks on x-axis
pylab.rcParams['ytick.major.size'] = 7      # size of ticks on y-axis
pylab.rcParams['lines.markersize'] = 10     # size of markers
pylab.rcParams['legend.numpoints'] = 1      # number of examples in legends

NAN = float('nan')


def minkowskiDist(v1, v2, p):
    """Computes the Minkowski distance between v1 and v2 using exponent p

    Args:
        v1 (list or 1D array): An ordered collection of n numbers,
        v1 (list or 1D array): Another ordered collection of n numbers
        p (float): The exponent used for the Minkowski distance

    Returns:
        The Minkowski distance between v1 and v2, [sum_i |v1[i]-v2[i]|^p]^(1/p)
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    abs_diffs = np.abs(v1 - v2)
    return np.sum(np.pow(abs_diffs, p)) ** (1 / p)


class Passenger(object):
    """Data structure representing a single example in the Titanic dataset"""

    FEATURE_NAMES = ('C1', 'C2', 'C3', 'age', 'male gender')

    def __init__(self, cabin, age, gender, survived, name):
        self.name = name
        self.featureVec = [0, 0, 0, age, gender]
        self.featureVec[cabin - 1] = 1
        self.label = survived
        self.cabin = cabin

    def distance(self, other):
        return minkowskiDist(self.featureVec, other.featureVec, 2)

    def getCabin(self):
        return self.cabin

    def getAge(self):
        return self.featureVec[3]

    def getGender(self):
        return self.featureVec[4]

    def getName(self):
        return self.name

    def getFeatures(self):
        return self.featureVec[:]

    def getLabel(self):
        return self.label


def createPassenger(line):
    """Create a passenger corresponding to one line of titanic data"""
    attributes = line.split(',')
    cabin, age, gender, survived = attributes[:4]
    name = attributes[4:]

    # clean data / convert strings to numbers
    cabin = int(cabin)
    age = float(age)
    gender = 1 if gender == 'M' else 0
    survived = 'Survived' if survived == '1' else 'Died'

    return Passenger(
        cabin=cabin, age=age, gender=gender, survived=survived, name=name)


def readTitanicData(fileName):
    """Create a collection of Passengers from the data in a file

    Args:
        fileName (str): the file to read

    Returns:
        A list of Passenger instances; each passenger has attributes given
        by one line of the file
    """
    with open(fileName) as f:
        examples = [createPassenger(line) for line in f]
    print('Finished processing', len(examples), 'passengers\n')
    return examples


def accuracy(truePos, falsePos, trueNeg, falseNeg):
    """Computes the accuracy given the true/false positives/negatives

    Args:
        truePos (float): The number of true positives
        falsePos (float): The number of false positives
        trueNeg (float): The number of true negatives
        falseNeg (float): The number of false negatives

    Returns:
        The accuracy as a float
    """
    numerator = truePos + trueNeg
    denominator = truePos + trueNeg + falsePos + falseNeg
    return numerator / denominator


def safeDivide(x, y):
    """Returns x / y if y is nonzero, else NaN"""
    try:
        return x / y
    except ZeroDivisionError:
        return NAN


def sensitivity(truePos, falseNeg):
    """Computes the sensitivity, the fraction of positives detected

    Args:
        truePos (float): The number of true positives
        falseNeg (float): The number of false negatives

    Returns:
        The sensitivity as a float
    """
    return safeDivide(truePos, truePos + falseNeg)


def specificity(trueNeg, falsePos):
    """Computes the sensitivity, the fraction of negatives detected

    Args:
        trueNeg (float): The number of true negatives
        falsePos (float): The number of false positives

    Returns:
        The specificity as a float
    """
    return safeDivide(trueNeg, trueNeg + falsePos)


def posPredVal(truePos, falsePos):
    """Computes the positive predictive value (PPV)

    The PPV is the fraction of reported positives that are true positives.

    Args:
        truePos (float): The number of true positives
        falsePos (float): The number of false positives

    Returns:
        The PPV as a float
    """
    return safeDivide(truePos, truePos + falsePos)


def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint=True):
    """Computes various performance statistics

    Args:
        truePos (float): The number of true positives
        falsePos (float): The number of false positives
        trueNeg (float): The number of true negatives
        falseNeg (float): The number of false negatives

    Returns:
        stats (tuple of four floats): contains (accuracy, sensitivity,
            specificity, positive predictive value), in this order
    """
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
    sens = sensitivity(truePos, falseNeg)
    spec = specificity(trueNeg, falsePos)
    ppv = posPredVal(truePos, falsePos)
    return accur, sens, spec, ppv


def split80_20(examples):
    """Partition the examples into groups containing 80% and 20% of the data

    Args:
        examples: a collection of arbitrary objects

    Returns:
        trainingSet: a list containing ~80% of the objects in examples
        testSet: a list containing the remaining objects in examples
    """
    sampleIndices = random.sample(range(len(examples)), len(examples) // 5)
    trainingSet, testSet = [], []
    for i, example in enumerate(examples):
        if i in sampleIndices:
            testSet.append(example)
        else:
            trainingSet.append(example)
    return trainingSet, testSet


def randomSplits(examples, method, numSplits, printStats=True):
    """Assess the method's performance on the examples using multiple splits

    Args:
        examples (collection of objects): data from which to draw the
            training and test sets
        method (function): a function such that method(trainingSet, testSet)
            returns a 4-tuple of (true positives, false positives,
            true negatives, false negatives), where trainingSet and testSet
            are disjoint subsets of the objects in examples
        numSplits (int): the number of train/test splits to use to compute
            performance measures
        printStats (bool): whether to print out the resulting performance
            statistics

    Returns:
        stats (tuple of four floats): contains (accuracy, sensitivity,
            specificity, positive predictive value), in this order
    """
    random.seed(0)

    # compute average {true, false} {positives, negatives} across all splits
    results = np.zeros(4)
    for t in range(numSplits):
        trainingSet, testSet = split80_20(examples)
        results += method(trainingSet, testSet)
    results /= numSplits

    # unpack individual results and compute performance stats
    truePos, falsePos, trueNeg, falseNeg = results
    accur, sens, spec, ppv = getStats(truePos, falsePos, trueNeg, falseNeg)

    if printStats:
        print_stat = lambda msg, val: print(' {} = {}'.format(
            msg, round(val, 3)))
        print_stat('Accuracy', accur)
        print_stat('Sensitivity', sens)
        print_stat('Specificity', spec)
        print_stat('Pos. Pred. Val.', ppv)

    return tuple(results)


def buildModel(examples, toPrint=True):
    """Creates and fits a logistic regression model

    Args:
        examples (collection of Passengers): the data on which to train
            the model
        toPrint (bool): whether to print the set of classes and the weights
            learned for each feature

    Returns:
        An sklearn LogististicRegression model fitted to the examples
    """
    featureVecs, labels = [], []
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
    model = LogisticRegression().fit(featureVecs, labels)
    if toPrint:
        # print the class names and feature weights
        print('model.classes_ =', model.classes_)
        for j, weight in enumerate(model.coef_[0]):
            print('   ', Passenger.FEATURE_NAMES[j], '=', weight)
    return model


def applyModel(model, testSet, label, threshold=0.5):
    """Computes model's predictions on the test set

    Args:
        model: a fitted sklearn classifier with the predict_proba() method
        testSet (collection of Passenger): a  objects
        label (str): label associated with "positives" (as opposed
            to "negatives")
        threshold (float): the probability above which a predicted probability
            should be considered a prediction of the positive label

    Returns:
        truePos (float): The number of true positives
        falsePos (float): The number of false positives
        trueNeg (float): The number of true negatives
        falseNeg (float): The number of false negatives
    """
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i, prob in enumerate(probs):
        # determine whether the predicted probability is > the threshold
        # to get the predicted class, and then determine whether this
        # prediction is correct or incorrect
        if prob[1] > threshold:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg


def train_test_logistic_regression(trainingData, testData, threshold=0.5):
    """Trains and evaluates a logistic regression model

    Args:
        trainingData (collection of Passengers): the examples on which
            to train the model
        testData (collection of Passengers): the examples on which to
            test the model
        threshold (float): the probability above which a predicted probability
            should be considered a prediction of the positive label

    Results:
        stats (tuple of four floats): contains (accuracy, sensitivity,
            specificity, positive predictive value), in this order
    """
    model = buildModel(trainingData, False)
    return applyModel(model, testData, 'Survived', threshold)


def plotROC(model, testSet, title):
    """Plots the ROC curve for the model evaluated on the test set

    Args:
        model: a fitted sklearn classifer
        testSet (collection of Passengers): data on which to evaluate the model
        title (str): a title for the plot

    Returns:
        The AUC as a float
    """
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg = \
            applyModel(model, testSet, 'Survived', p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = sklearn.metrics.auc(xVals, yVals, True)

    pylab.plot(xVals, yVals)
    pylab.plot([0, 1], [0, 1])
    title = title + '\nAUROC = ' + str(round(auroc, 3))
    pylab.title(title)
    pylab.xlabel('1 - specificity')
    pylab.ylabel('Sensitivity')
    pylab.show()

    return auroc


def run_training_and_eval():
    """main function"""
    random.seed(0)
    examples = readTitanicData('TitanicPassengers.txt')

    randomSplits(examples, train_test_logistic_regression, numSplits=10)

    trainingSet, testSet = split80_20(examples)
    model = buildModel(trainingSet, testSet)
    plotROC(model, testSet, 'ROC for Predicting Survival, 1 Split')


if __name__ == '__main__':
    run_training_and_eval()
