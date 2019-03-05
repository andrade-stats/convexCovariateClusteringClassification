
import numpy
import scipy.misc
import scipy.optimize
import time

def gradientApprox(func, X, i, j):
    epsilon = 0.0001
    Xplus = numpy.copy(X)
    Xplus[i,j] += epsilon
    Xminus = numpy.copy(X)
    Xminus[i,j] -= epsilon
    grad = (func(Xplus) - func(Xminus)) / (2.0 * epsilon)
    print("gradient approximation = ", grad)
    return


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



def precalculationOfAllQsFast(edgesZ, edgesU, allAdjacentNodes):
    NUMBER_OF_COVARIATES = edgesZ.shape[0]
    NUMBER_OF_CLASSES = edgesZ.shape[2]
    
    qUp = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
    qAll = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i]
        if len(adjacentNodes) > 0:
            q = edgesZ[i, adjacentNodes] + edgesU[i, adjacentNodes]
            qAll += numpy.sum(numpy.square(q))
            qUp[i] = numpy.sum(q, axis = 0)
     
    return qUp, qAll



def fastRegulationPartForObjectiveValue(qUp, qAll, edgesZ, edgesU, B, beta, allAdjacentNodes):
    NUMBER_OF_COVARIATES = edgesZ.shape[0]
    
    weightedBSquareTerm = 0.0
    bTimesqUpTerm = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i]
        bTimesqUpTerm += numpy.dot(qUp[i],B[:,i])
        weightedBSquareTerm += len(adjacentNodes) * numpy.dot(B[:,i],B[:,i])
        
    reg = qAll - 2.0 * bTimesqUpTerm + weightedBSquareTerm
    
    return reg




# updated
def slowRegulationPartForObjectiveValue(edgesZ, edgesU, B, beta, allAdjacentNodes):
    NUMBER_OF_COVARIATES = edgesZ.shape[0]
    
    regSlowCorrect = 0
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i]
        for j in adjacentNodes:
            diffM = edgesZ[i,j] - B[:,i] + edgesU[i,j]
            regSlowCorrect += numpy.sum(numpy.square(diffM))
    
    # print "reg (correct calculation) = ", regSlowCorrect
    
    return regSlowCorrect
 



# calculates the negative log likelihood + regularization term from ADMM
# reading checked
def getObjValueNew(qUp, qAll, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION):
    NUMBER_OF_COVARIATES = edgesZ.shape[0]
    assert(B.shape[1] == NUMBER_OF_COVARIATES)
    assert(type(dataFeatures) is numpy.ndarray and type(dataLabels) is numpy.ndarray)
    assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
    
    totalNegLogProb = getTotalNegLogProb(dataFeatures, dataLabels, B, beta)
    
    start_time_objval = time.time() # measure time of necessary for regularizer
           
    if VERSION == "SLOW_VERSION":
        assert(False)
        reg = slowRegulationPartForObjectiveValue(edgesZ, edgesU, B, beta, allAdjacentNodes)
    else:
        assert(VERSION == "FAST_VERSION")
        reg = fastRegulationPartForObjectiveValue(qUp, qAll, edgesZ, edgesU, B, beta, allAdjacentNodes)
    
    
    for i in range(NUMBER_OF_COVARIATES):
        diffM = singleZ[i] - B[:,i] + singleU[i] 
        reg += numpy.sum(numpy.square(diffM))
    
    regRuntime = (time.time() - start_time_objval) 
    
    return totalNegLogProb + 0.5 * rho * reg, regRuntime



# checked
def getAllProbs(dataFeatures, B, beta):
    
    allUnnormalizedLogProbs = numpy.dot(dataFeatures, B.transpose()) + beta
    allLogSums = scipy.misc.logsumexp(allUnnormalizedLogProbs, axis = 1)
  
    logProbs = (allUnnormalizedLogProbs.transpose() - allLogSums).transpose()
    return numpy.exp(logProbs)


# updated
def slowRegulationPartForGradient(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
 
    # add part from regularizer
    gradientBfromRegSlow = numpy.zeros_like(B)
    for i in range(NUMBER_OF_COVARIATES):
        gradientBfromRegSlow[:, i] += B[:,i] - singleZ[i] - singleU[i] 
        adjacentNodes = allAdjacentNodes[i]
        for j in adjacentNodes:
            gradientBfromRegSlow[:, i] += B[:,i] - edgesZ[i,j] - edgesU[i,j]
            
    return gradientBfromRegSlow



# updated
# reading checked
def getGradientNew(columnWeightsForB, qUp, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION):
    NUMBER_OF_CLASSES = B.shape[0]
    NUMBER_OF_COVARIATES = B.shape[1]
    assert(NUMBER_OF_CLASSES < NUMBER_OF_COVARIATES)
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
          
    start_time_grad = time.time() # measure time of necessary for regularizer 
   
    bSum = numpy.sum(B, axis = 1)
    assert(bSum.shape[0] == NUMBER_OF_CLASSES)
    assert(len(bSum.shape) == 1)
      
    # SLOW VERSION:
    if VERSION == "SLOW_VERSION":
        assert(False)
        gradientBfromReg = slowRegulationPartForGradient(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes)
    else:
        assert(VERSION == "FAST_VERSION")
        # NEW FAST VERSION:
        qReadyForGradReg = (qUp + singleZ + singleU).transpose()
        gradientBfromReg = numpy.multiply(B, columnWeightsForB) - qReadyForGradReg
        
    # print "gradient check sum (correct) = ", numpy.sum(gradientBfromRegOldCorrect)
    # print "gradient check sum (fast) = ", numpy.sum(gradientBfromReg)
    # assert(False)
    
    gradientB += rho * gradientBfromReg
    regRuntime = (time.time() - start_time_grad)
    
    return gradientB, gradientBeta, regRuntime



def convertToOneVec(B, beta):
    numberOfClasses = B.shape[0]
    asOneMatrix = numpy.concatenate((B, beta.reshape((numberOfClasses,1))), axis = 1)
    return asOneMatrix.ravel()



def getGradientForDebugTest(dataFeatures, dataLabels, B, beta):
    NUMBER_OF_CLASSES = B.shape[0]
    NUMBER_OF_COVARIATES = B.shape[1]
    assert(NUMBER_OF_CLASSES < NUMBER_OF_COVARIATES)
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
    return -1 * gradientB




    
def gradientEstimate(f, i, j, x):
    EPSILON = 0.0001
    xPlus = numpy.copy(x)
    xPlus[i,j] += EPSILON
    xMinus = numpy.copy(x)
    xMinus[i,j] -= EPSILON
    return (f(xPlus) - f(xMinus)) / (2.0 * EPSILON)


# updated
# B0, beta0 are the initial guesses
def optimizeLBFGSNew(dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B0, beta0, allAdjacentNodes, MAX_ITERATIONS):
    NUMBER_OF_CLASSES = B0.shape[0]
    NUMBER_OF_COVARIATES = B0.shape[1]
    assert(beta0.shape[1] == NUMBER_OF_CLASSES)

    TIME_MEASURE_COUNT = [0]
    GRADIENT_TIME_MEASURE_TOTAL = [0.0]
    FUNCVALUE_TIME_MEASURE_TOTAL = [0.0]

    # VERSION = "SLOW_VERSION"
    VERSION = "FAST_VERSION"
        
    if VERSION == "SLOW_VERSION":
        qUp, qAll = None, None
        columnWeightsForB = None
    else:
        start_time_bStepPrecalculations = time.time()
        qUp, qAll = precalculationOfAllQsFast(edgesZ, edgesU, allAdjacentNodes)
        
        columnWeightsForB = numpy.zeros(NUMBER_OF_COVARIATES)
        for i in range(NUMBER_OF_COVARIATES):
            columnWeightsForB[i] = len(allAdjacentNodes[i]) + 1.0
            
        # r = precalculationOfR(pairedZ, singleZ, pairedU, singleU)
        # print "finished bStepPrecalculations in = ", ((time.time() - start_time_bStepPrecalculations) / (60.0))
        
    def funcAndGradForLBFG(X):
        B, beta = convertBack(X)
        objValue, regRuntimeFuncValue = getObjValueNew(qUp, qAll, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION)
        gradB, gradBeta, regRuntimeGradient = getGradientNew(columnWeightsForB, qUp, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION) 
        
        FUNCVALUE_TIME_MEASURE_TOTAL[0] += regRuntimeFuncValue
        GRADIENT_TIME_MEASURE_TOTAL[0] += regRuntimeGradient
        TIME_MEASURE_COUNT[0] += 1
    
        grad = convertToOneVec(gradB, gradBeta)
        # print "-> objValue = ", objValue
        # print "grad = ", grad
        # assert(False)
        
        return (objValue, grad)
    
    def convertBack(X):
        Xview = X.view()
        Xview.shape = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES + 1)
        B = Xview[:,0:NUMBER_OF_COVARIATES]
        beta = Xview[:,NUMBER_OF_COVARIATES]
        beta = beta.reshape((1, NUMBER_OF_CLASSES))
        return B, beta
    
    # runGradientCheck(dataFeatures, dataLabels, pairedZ, singleZ, pairedU, singleU, rho, B0, beta0)
    
    # print "START B-STEP"
    # start_time_bStep = time.time() 
    
    bestX, objValue, otherInfo = scipy.optimize.fmin_l_bfgs_b(func = funcAndGradForLBFG, x0 = convertToOneVec(B0, beta0), maxiter = MAX_ITERATIONS)
    bestB, bestBeta = convertBack(bestX)
    # print "b-step runtime (sec)" , (time.time() - start_time_bStep) 
    
    averageFuncValueTime = FUNCVALUE_TIME_MEASURE_TOTAL[0] / float(TIME_MEASURE_COUNT[0])
    averageGradientTime =  GRADIENT_TIME_MEASURE_TOTAL[0] / float(TIME_MEASURE_COUNT[0])
    totalTime = FUNCVALUE_TIME_MEASURE_TOTAL[0] + GRADIENT_TIME_MEASURE_TOTAL[0]
    # print "number of gradient evaluations = ", TIME_MEASURE_COUNT[0]
    # print "total time regularizer part in gradient + funcValue (sec) = ", totalTime
    # print "average time for regularizer part in funcValue (sec) = ", averageFuncValueTime
    # print "average time for regularizer part in gradient (sec) = ", averageGradientTime
    # assert(False)
    
    return bestB, bestBeta

