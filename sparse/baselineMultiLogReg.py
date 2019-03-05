
# ***************************************************
# SYMMETRIC VERSION
#  implements multinomial logisitic regression with l2 penalty using c parameter vectors, 
#  where c is the number of classes
# ***************************************************


import numpy
import scipy.misc
import scipy.optimize
import sklearn.metrics
import crossvalidation
import helper


    
def splitIntoValidAndTest(allTestDataAsMatrixOrig, allTestLabelsOrig):
    assert(allTestDataAsMatrixOrig.shape[0] == allTestLabelsOrig.shape[0])
    n = allTestDataAsMatrixOrig.shape[0]
    
    allIndices = numpy.arange(n)
    numpy.random.shuffle(allIndices)
    nHalf = int(n / 2)
    validIndices = allIndices[0:nHalf]
    testIndices = allIndices[nHalf:n]
    
    print("real validation and test size = ", nHalf)
    
    allValidDataAsMatrix = allTestDataAsMatrixOrig[validIndices]
    allValidLabels = allTestLabelsOrig[validIndices]
    allTestDataAsMatrix = allTestDataAsMatrixOrig[testIndices]
    allTestLabels = allTestLabelsOrig[testIndices]

    return allValidDataAsMatrix, allValidLabels, allTestDataAsMatrix, allTestLabels


def predictLabels(dataFeatures, B, beta):
    assert(type(dataFeatures) is numpy.ndarray)
    assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
    
    allUnnormalizedLogProbs = numpy.dot(dataFeatures, B.transpose()) + beta
    return numpy.argmax(allUnnormalizedLogProbs, axis = 1)
    
    

# returns unpenalized negative log likelihood
def getTotalNegLogProb(dataFeatures, dataLabels, B, beta):
    assert(type(dataFeatures) is numpy.ndarray and type(dataLabels) is numpy.ndarray)
    assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
    
    allUnnormalizedLogProbs = numpy.dot(dataFeatures, B.transpose()) + beta
    
    allLogSums = scipy.misc.logsumexp(allUnnormalizedLogProbs, axis = 1)
    allSelectedULP = allUnnormalizedLogProbs[numpy.arange(dataLabels.shape[0]), dataLabels]
    
    totalNegLogProb = - numpy.sum(allSelectedULP) + numpy.sum(allLogSums)
    return totalNegLogProb


 

# calculates the negative log likelihood + l2 regularization
# NEW reading checked
def getObjValue(dataFeatures, dataLabels, B, beta, lambdaParam):
    assert(type(dataFeatures) is numpy.ndarray and type(dataLabels) is numpy.ndarray)
    assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
    
    totalNegLogProb = getTotalNegLogProb(dataFeatures, dataLabels, B, beta)
    reg = numpy.sum(numpy.square(B))
        
    return totalNegLogProb + lambdaParam * reg


# checked
def getAllProbs(dataFeatures, B, beta):
    
    allUnnormalizedLogProbs = numpy.dot(dataFeatures, B.transpose()) + beta
    allLogSums = scipy.misc.logsumexp(allUnnormalizedLogProbs, axis = 1)
  
    logProbs = (allUnnormalizedLogProbs.transpose() - allLogSums).transpose()
    return numpy.exp(logProbs)




# updated
# NEW reading checked
def getGradient(dataFeatures, dataLabels, B, beta, lambdaParam):
    NUMBER_OF_CLASSES = B.shape[0]
    NUMBER_OF_COVARIATES = B.shape[1]
    assert(beta.shape[0] == 1 and beta.shape[1] == NUMBER_OF_CLASSES)
    assert(type(dataFeatures) is numpy.ndarray and type(dataLabels) is numpy.ndarray)
    assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
    
    # rows = samples, columns = class-labels
    weightFactorsForEachSample = getAllProbs(dataFeatures, B, beta)
        
    # for each data sample
    for i in range(dataFeatures.shape[0]):
        label = dataLabels[i]
        weightFactorsForEachSample[i, label] -= 1.0
    
   
    gradientB = numpy.dot(weightFactorsForEachSample.transpose(), dataFeatures)
    gradientBeta = numpy.sum(weightFactorsForEachSample, axis = 0)
       
    gradientBfromReg = 2.0 * lambdaParam * B
    gradientB += gradientBfromReg
    
    return gradientB, gradientBeta



def convertToOneVec(B, beta):
    numberOfClasses = B.shape[0]
    asOneMatrix = numpy.concatenate((B, beta.reshape((numberOfClasses,1))), axis = 1)
    return asOneMatrix.ravel()

# updated
# B0, beta0 are the initial guesses
def optimizeLBFGS(dataFeatures, dataLabels, B0, beta0, lambdaParam, MAX_LBFGS_ITERATIONS):
    
    NUMBER_OF_CLASSES = B0.shape[0]
    NUMBER_OF_COVARIATES = B0.shape[1]
    assert(beta0.shape[1] == NUMBER_OF_CLASSES)


    def funcAndGradForLBFG(X):
        B, beta = convertBack(X)
        objValue = getObjValue(dataFeatures, dataLabels, B, beta, lambdaParam)
        gradB, gradBeta = getGradient(dataFeatures, dataLabels, B, beta, lambdaParam)
        grad = convertToOneVec(gradB, gradBeta)
        
        return (objValue, grad)
    
    def convertBack(X):
        Xview = X.view()
        Xview.shape = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES + 1)
        B = Xview[:,0:NUMBER_OF_COVARIATES]
        beta = Xview[:,NUMBER_OF_COVARIATES]
        beta = beta.reshape((1, NUMBER_OF_CLASSES))
        return B, beta
    
    # runGradientCheck(dataFeatures, dataLabels, pairedZ, singleZ, pairedU, singleU, rho, B0, beta0)
     
    bestX, objValue, otherInfo = scipy.optimize.fmin_l_bfgs_b(func = funcAndGradForLBFG, x0 = convertToOneVec(B0, beta0),  maxiter=MAX_LBFGS_ITERATIONS)
    bestB, bestBeta = convertBack(bestX)
    
    # print "objValue = ", objValue
    
    return bestB, bestBeta, objValue


def evaluate(evalData, trueLabels, B, beta):
    predictedLabels = predictLabels(evalData, B, beta)
    microScore = sklearn.metrics.f1_score(y_true = trueLabels, y_pred = predictedLabels, average='micro')
    macroScore = sklearn.metrics.f1_score(y_true = trueLabels, y_pred = predictedLabels, average='macro')
    accuracy = sklearn.metrics.accuracy_score(y_true = trueLabels, y_pred = predictedLabels)
    return microScore, macroScore, accuracy

# checked
def trainAndTestFuncSimple(allParams):
    try:
        trainCovariates, trainLabels, testCovariates, testLabels, lambdaParam, evalCriteria = allParams
        NUMBER_OF_COVARIATES = trainCovariates.shape[1]
        NUMBER_OF_CLASSES = numpy.max(trainLabels) + 1
        assert(numpy.max(trainLabels) == numpy.max(testLabels))
        B0 = numpy.random.normal(size = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES))
        beta0 = numpy.random.normal(size = (1,NUMBER_OF_CLASSES))
        MAX_LBFGS_ITERATIONS = 15000 # default value
        learnedB, learnedBeta, _ = optimizeLBFGS(trainCovariates, trainLabels, B0, beta0, lambdaParam, MAX_LBFGS_ITERATIONS)
         
        if evalCriteria == "logProb":
            return - getTotalNegLogProb(testCovariates, testLabels, learnedB, learnedBeta)
        else:
            return crossvalidation.eval(testLabels, predictLabels(testCovariates, learnedB, learnedBeta), evalCriteria)
 
    except (KeyboardInterrupt, Exception):
        print("KEYBOARD INTERRUPT OR ERROR")

# new version of trainAndTestFunc:
def trainValidAndTestFunc(allParams):
    try:
        trainCovariates, trainLabels, validCovariates, validLabels, testCovariates, testLabels, lambdaParam, evalCriteria = allParams
        NUMBER_OF_COVARIATES = trainCovariates.shape[1]
        NUMBER_OF_CLASSES = numpy.max(trainLabels) + 1
        assert(numpy.max(trainLabels) == numpy.max(testLabels))
        B0 = numpy.random.normal(size = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES))
        beta0 = numpy.random.normal(size = (1,NUMBER_OF_CLASSES))
        MAX_LBFGS_ITERATIONS = 15000 # default value
        learnedB, learnedBeta, _ = optimizeLBFGS(trainCovariates, trainLabels, B0, beta0, lambdaParam, MAX_LBFGS_ITERATIONS)
        
        if evalCriteria == "logProb":
            return - getTotalNegLogProb(testCovariates, testLabels, learnedB, learnedBeta)
        else:
            assert(evalCriteria == "accuracy")
            validAccuracy = crossvalidation.eval(validLabels, predictLabels(validCovariates, learnedB, learnedBeta), evalCriteria)
            testAccuracy = crossvalidation.eval(testLabels, predictLabels(testCovariates, learnedB, learnedBeta), evalCriteria)
            return validAccuracy, testAccuracy
        
    except (KeyboardInterrupt, Exception):
        print("KEYBOARD INTERRUPT OR ERROR")


# checked
# new: returns logProb of cross-validation with best parameter
def runLogisticRegressionCVnew(allCovariatesAsMatrix, allLabels, evalCriteria):
    assert(allLabels.shape[0] == allCovariatesAsMatrix.shape[0])
    TRAINING_DATA_SIZE = allLabels.shape[0]
    NUMBER_OF_LABELS = numpy.max(allLabels) + 1
    
    if (TRAINING_DATA_SIZE / (5 * NUMBER_OF_LABELS)) >= 2:
        print("USE 5 FOLDS FOR CV")
        NUMBER_OF_FOLDS = 5
    else:
        print("WARNING NEED TO REDUCE FOLDS TO 2 FOR CV")
        NUMBER_OF_FOLDS = 2
        
        
    allSigmaValuesExp = numpy.arange(-3, 2, 0.5)
    allSigmaValues = [10 ** expI for expI in allSigmaValuesExp]
    
    print("test the following sigma values = ", allSigmaValues)
    allLambdaValuesToTest = [1.0 / (2.0 * sigma * sigma) for sigma in allSigmaValues]
    
    allResults, _ = crossvalidation.runCV(allCovariatesAsMatrix, allLabels, trainAndTestFuncSimple, allLambdaValuesToTest, evalCriteria, NUMBER_OF_FOLDS)
    
    bestParamId = numpy.argmax(allResults)
    bestLambdaParam = allLambdaValuesToTest[bestParamId]
    
    print("bestLambdaParam = ", bestLambdaParam) 
    return bestLambdaParam, numpy.max(allResults)

def estimateHeldOutPerformance(allCovariatesAsMatrix, allLabels, evalCriteria, lambdaParam):
    assert(allLabels.shape[0] == allCovariatesAsMatrix.shape[0])
    TRAINING_DATA_SIZE = allLabels.shape[0]
    NUMBER_OF_LABELS = numpy.max(allLabels) + 1
    
    if (TRAINING_DATA_SIZE / (5 * NUMBER_OF_LABELS)) >= 2:
        print("USE 5 FOLDS FOR CV")
        NUMBER_OF_FOLDS = 5
    else:
        print("WARNING NEED TO REDUCE FOLDS TO 2 FOR CV")
        NUMBER_OF_FOLDS = 2
     
    allHyperparameters = [lambdaParam] 
    allResults, allResultsSD = crossvalidation.runCV(allCovariatesAsMatrix, allLabels, trainAndTestFuncSimple, allHyperparameters, evalCriteria, NUMBER_OF_FOLDS)
    assert(allResults.shape[0] == 1)
    assert(allResultsSD.shape[0] == 1)
    return allResults[0], allResultsSD[0]


def evalModelOnTrainAndTestDataNew(origDataFeaturesTrain, dataLabelsTrain, sortedClusters, origDataFeaturesTest, dataLabelsTest, sigma):
    
    dataFeaturesTrain = helper.projectData(origDataFeaturesTrain, sortedClusters)
    dataFeaturesTest = helper.projectData(origDataFeaturesTest, sortedClusters)
    
    lambdaParam = 1.0 / (2.0 * sigma * sigma)
    
    trainAccuracy, testAccuracy = trainValidAndTestFunc((dataFeaturesTrain, dataLabelsTrain, dataFeaturesTrain, dataLabelsTrain, dataFeaturesTest, dataLabelsTest, lambdaParam, "accuracy"))
    return trainAccuracy, testAccuracy

    
    
def evalModelOnTrainValidAndTestData(origDataFeaturesTrain, dataLabelsTrain, sortedClusters, origDataFeaturesValid, dataLabelsValid, origDataFeaturesTest, dataLabelsTest):
    
    dataFeaturesTrain = helper.projectData(origDataFeaturesTrain, sortedClusters)
    dataFeaturesValid = helper.projectData(origDataFeaturesValid, sortedClusters)
    dataFeaturesTest = helper.projectData(origDataFeaturesTest, sortedClusters)
    
    bestLambdaParam, _ = runLogisticRegressionCVnew(dataFeaturesTrain, dataLabelsTrain, "logProb")
    
    validAccuracy, testAccuracy = trainValidAndTestFunc((dataFeaturesTrain, dataLabelsTrain, dataFeaturesValid, dataLabelsValid, dataFeaturesTest, dataLabelsTest, bestLambdaParam, "accuracy"))
    return validAccuracy, testAccuracy    


    