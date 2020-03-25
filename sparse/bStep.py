
import numpy
import scipy.misc
import scipy.optimize
import time
import helper


def gradientApprox(func, X, i, j):
    epsilon = 0.0001
    Xplus = numpy.copy(X)
    Xplus[i,j] += epsilon
    Xminus = numpy.copy(X)
    Xminus[i,j] -= epsilon
    grad = (func(Xplus) - func(Xminus)) / (2.0 * epsilon)
    print("gradient approximation = ", grad)
    return


# def runGradientCheck(dataFeatures, dataLabels, pairedZ, singleZ, pairedU, singleU, rho, B0, beta0):
#     
#     def func(B):
#         objValue, _ = getObjValueNew(dataFeatures, dataLabels, pairedZ, singleZ, pairedU, singleU, rho, B, beta0)
#         return objValue
#     
#     
#     i = 0
#     j = 2
#     gradientApprox(func, B0, i, j)
# 
#     gradB, gradBeta = getGradientNew(dataFeatures, dataLabels, pairedZ, singleZ, pairedU, singleU, rho, B0, beta0)
#      
#     print "my calculated gradient = "
#     print gradB[i,j]    
# 
#     assert(False)
#     return


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

    # add squared loss penalty to ensure identifiability and strict convexness 
    # B_DIM = B.shape[0] * B.shape[1]
    totalNegLogProb += helper.LAMBDA * numpy.sum(numpy.square(B))

    return totalNegLogProb


# def precalculationOfR(pairedZ, singleZ, pairedU, singleU):
#     NUMBER_OF_COVARIATES = pairedZ.shape[0]
#     NUMBER_OF_CLASSES = pairedZ.shape[2]
#     
#     r = numpy.zeros(shape = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES))
#     for i in xrange(NUMBER_OF_COVARIATES):
#         r[:, i] -= singleZ[i] + singleU[i] 
#         for j in xrange(NUMBER_OF_COVARIATES):
#             if j < i:
#                 r[:, i] += pairedZ[j,i] + pairedU[j,i]
#             elif j > i:
#                 r[:, i] -= pairedZ[i,j] + pairedU[i,j]
#     
#     return r


# def precalculationOfAllQs(pairedZ, pairedU):
#     NUMBER_OF_COVARIATES = pairedZ.shape[0]
#     assert(NUMBER_OF_COVARIATES == pairedZ.shape[1])
#     NUMBER_OF_CLASSES = pairedZ.shape[2]
#    
#     qUp = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
#     
#     # note that qUp[NUMBER_OF_COVARIATES-1] remains 0
#     for i in xrange(NUMBER_OF_COVARIATES):
#         for j in xrange(i+1, NUMBER_OF_COVARIATES, 1):
#             qUp[i] += pairedZ[i,j] + pairedU[i,j]
#     qUp = qUp.transpose()
# 
#     qDown = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
#     
#     # note that qDown[0] remains 0 
#     for j in xrange(1, NUMBER_OF_COVARIATES, 1): 
#         for i in xrange(j):
#             qDown[j] += pairedZ[i,j] + pairedU[i,j]
#     qDown = qDown.transpose()
#      
#     qq = 0.0
#     for i in xrange(NUMBER_OF_COVARIATES):
#         for j in xrange(i+1, NUMBER_OF_COVARIATES, 1):
#             qq += numpy.sum(numpy.square(pairedZ[i,j] + pairedU[i,j]))
#     # qq = numpy.sum(numpy.square(pairedZ + pairedU))
#     
#     return qUp, qDown, qq


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
    
    # add squared loss penalty to ensure identifiability and strict convexness 
    gradientB += helper.LAMBDA * 2.0 * B

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



# NUMBER_OF_CLASSES = 3
# NUMBER_OF_COVARIATES = 2
# 
# fullHessian = numpy.zeros((NUMBER_OF_CLASSES * NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES * NUMBER_OF_COVARIATES))
# for i in xrange(NUMBER_OF_CLASSES):
#     print "** i = ", i
#     for j in xrange(i,NUMBER_OF_CLASSES, 1):
#         print "j = ", j
#         fullHessian[(i * NUMBER_OF_COVARIATES) : ((i+1) * NUMBER_OF_COVARIATES), (j * NUMBER_OF_COVARIATES) : ((j+1) * NUMBER_OF_COVARIATES)] = numpy.ones((NUMBER_OF_COVARIATES,NUMBER_OF_COVARIATES)) * (i+j)
#         fullHessian[(j * NUMBER_OF_COVARIATES) : ((j+1) * NUMBER_OF_COVARIATES), (i * NUMBER_OF_COVARIATES) : ((i+1) * NUMBER_OF_COVARIATES)] = numpy.ones((NUMBER_OF_COVARIATES,NUMBER_OF_COVARIATES)) * (i+j)
# 
# print "fullHessian = "
# print fullHessian


    
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

#     def f(B):
#         logProbsAllClasses = numpy.log(getAllProbs(dataFeatures, B, beta0))
#         
#         dataLogProbs = 0.0
#         for i in xrange(logProbsAllClasses.shape[0]):
#             dataLogProbs += logProbsAllClasses[i,dataLabels[i]]
#         
#         return dataLogProbs

   
#     firstClassId = 2
#     secondClassId = 2
#     
#     outputFeatureId = 6
#     inputFeatureId = 6
#        
#     sigma = 1.0
#    
#     def f(B):
#         return getGradientForDebugTest(dataFeatures, dataLabels, B, beta0)[firstClassId,outputFeatureId]
#      
#     print "gradient estimate = ", gradientEstimate(f, secondClassId, inputFeatureId, B0)
#     
#     fullHessian = getFullHessian(dataFeatures, dataLabels, B0, beta0, sigma)
#     hessianBlock = fullHessian[(firstClassId * NUMBER_OF_COVARIATES) : ((firstClassId+1) * NUMBER_OF_COVARIATES), (secondClassId * NUMBER_OF_COVARIATES) : ((secondClassId+1) * NUMBER_OF_COVARIATES)]
#     print "hessian[input,output] = ", hessianBlock[inputFeatureId, outputFeatureId]
#     
#     s, logDetH = numpy.linalg.slogdet(fullHessian * -1)
#     print "s = ", s
#     # assert(s == -1)
#     print "log |H| = ", logDetH
#     print "lowest eigenvalues = ", numpy.linalg.eigvalsh(fullHessian * -1)[0:50]
#     assert(False)



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
        
        # assert(False)

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





# # qUp = sum_{edges} (z_edge + (1/rho)*u_edge)
# def precalculationOfAllQsFast_unscaled(edgesZ, edgesU, rho, allAdjacentNodes):
#     NUMBER_OF_COVARIATES = edgesZ.shape[0]
#     NUMBER_OF_CLASSES = edgesZ.shape[2]
#     
#     # qUpPlus = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
#     qUpMinus = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
#     zAll = 0.0
#     for i in xrange(NUMBER_OF_COVARIATES):
#         adjacentNodes = allAdjacentNodes[i]
#         q = edgesZ[i, adjacentNodes] + (1.0 / rho) * edgesU[i, adjacentNodes]
#         zAll += numpy.sum(numpy.square(edgesZ[i, adjacentNodes]))
#         # qUpPlus[i] = numpy.sum(q, axis = 0)
#      
#         qMinus = edgesZ[i, adjacentNodes] - (1.0 / rho) * edgesU[i, adjacentNodes]
#         qUpMinus[i] = numpy.sum(qMinus, axis = 0)
#         
#     return qUpMinus, zAll
# 
# 
# def fastRegulationPartForObjectiveValue_unscaled(qUpMinus, zAll, edgesZ, edgesU, B, beta, allAdjacentNodes):
#     NUMBER_OF_COVARIATES = edgesZ.shape[0]
#     
#     weightedBSquareTerm = 0.0
#     bTimesqUpTerm = 0.0
#     for i in xrange(NUMBER_OF_COVARIATES):
#         adjacentNodes = allAdjacentNodes[i]
#         bTimesqUpTerm += numpy.dot(qUpMinus[i],B[:,i])
#         weightedBSquareTerm += len(adjacentNodes) * numpy.dot(B[:,i],B[:,i])
#         
#     reg = zAll - 2.0 * bTimesqUpTerm + weightedBSquareTerm    
#     return reg


# 
# # updated
# def slowRegulationPartForObjectiveValue_unscaled(edgesZ, B, beta, allAdjacentNodes):
#     NUMBER_OF_COVARIATES = edgesZ.shape[0]
#     
#     regSlowCorrect = 0
#     for i in xrange(NUMBER_OF_COVARIATES):
#         adjacentNodes = allAdjacentNodes[i]
#         for j in adjacentNodes:
#             diffM = edgesZ[i,j] - B[:,i]
#             regSlowCorrect += numpy.sum(numpy.square(diffM))
#     
#     # print "reg (correct calculation) = ", regSlowCorrect
#     
#     return regSlowCorrect
# 
# def getObjValueNew_unscaled(qUpMinus, zAll, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION):
#     NUMBER_OF_COVARIATES = edgesZ.shape[0]
#     assert(B.shape[1] == NUMBER_OF_COVARIATES)
#     assert(type(dataFeatures) is numpy.ndarray and type(dataLabels) is numpy.ndarray)
#     assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
#     
#     totalNegLogProb = getTotalNegLogProb(dataFeatures, dataLabels, B, beta)
#     
#     start_time_objval = time.time() # measure time of necessary for regularizer
#            
#     if VERSION == "SLOW_VERSION":
#         reg = slowRegulationPartForObjectiveValue_unscaled(edgesZ, B, beta, allAdjacentNodes)
#     else:
#         assert(VERSION == "FAST_VERSION")
#         reg = fastRegulationPartForObjectiveValue_unscaled(qUpMinus, zAll, edgesZ, edgesU, B, beta, allAdjacentNodes)
#     
#     
#     for i in xrange(NUMBER_OF_COVARIATES):
#         diffM = singleZ[i] - B[:,i] 
#         reg += numpy.sum(numpy.square(diffM))
#     
#     
#     unscaledLinearPart = 0.0
#     for i in xrange(NUMBER_OF_COVARIATES):
#         unscaledLinearPart += numpy.dot(B[:,i], singleU[i]) 
#         
#     for i in xrange(NUMBER_OF_COVARIATES):
#         adjacentNodes = allAdjacentNodes[i]
#         for j in adjacentNodes:
#             unscaledLinearPart += numpy.dot(B[:,i], edgesU[i,j])
#         
#     regRuntime = (time.time() - start_time_objval) 
#     
#     return totalNegLogProb + 0.5 * rho * reg + unscaledLinearPart, regRuntime
# 
# 
# def slowRegulationPartForGradient_unscaled(rho, edgesZ, singleZ, B, allAdjacentNodes):
#     NUMBER_OF_COVARIATES = B.shape[1]
#  
#     # add part from regularizer
#     gradientBfromRegSlow = numpy.zeros_like(B)
#     for i in xrange(NUMBER_OF_COVARIATES):
#         gradientBfromRegSlow[:, i] += B[:,i] - singleZ[i]
#         adjacentNodes = allAdjacentNodes[i]
#         for j in adjacentNodes:
#             gradientBfromRegSlow[:, i] += B[:,i] - edgesZ[i,j]
#             
#     return gradientBfromRegSlow


# def getGradientNew_unscaled(columnWeightsForB, qUpMinus, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION):
#     NUMBER_OF_CLASSES = B.shape[0]
#     NUMBER_OF_COVARIATES = B.shape[1]
#     assert(NUMBER_OF_CLASSES < NUMBER_OF_COVARIATES)
#     assert(beta.shape[0] == 1 and beta.shape[1] == NUMBER_OF_CLASSES)
#     assert(type(dataFeatures) is numpy.ndarray and type(dataLabels) is numpy.ndarray)
#     assert(type(B) is numpy.ndarray and type(beta) is numpy.ndarray)
#     
#     # rows = samples, columns = class-labels
#     weightFactorsForEachSample = getAllProbs(dataFeatures, B, beta)
#         
#     # for each data sample
#     for i in xrange(dataFeatures.shape[0]):
#         label = dataLabels[i]
#         weightFactorsForEachSample[i, label] -= 1.0
#     
#    
#     gradientB = numpy.dot(weightFactorsForEachSample.transpose(), dataFeatures)
#     gradientBeta = numpy.sum(weightFactorsForEachSample, axis = 0)
#           
#     start_time_grad = time.time() # measure time of necessary for regularizer 
#    
#     bSum = numpy.sum(B, axis = 1)
#     assert(bSum.shape[0] == NUMBER_OF_CLASSES)
#     assert(len(bSum.shape) == 1)
#       
#     # SLOW VERSION:
#     if VERSION == "SLOW_VERSION":
#         # assert(False)
#         gradientBfromReg = slowRegulationPartForGradient_unscaled(rho, edgesZ, singleZ, B, allAdjacentNodes)
#     else:
#         assert(False)
#         assert(VERSION == "FAST_VERSION")
#         # NEW FAST VERSION:
#         qReadyForGradReg = (qUpMinus + singleZ - (1.0 / rho) * singleU).transpose()
#         gradientBfromReg = numpy.multiply(B, columnWeightsForB) - qReadyForGradReg
#         
#     # print "gradient check sum (correct) = ", numpy.sum(gradientBfromRegOldCorrect)
#     # print "gradient check sum (fast) = ", numpy.sum(gradientBfromReg)
#     # assert(False)
#     
#     gradientBfromUnscaledLinearPart = numpy.zeros_like(B)
#     for i in xrange(NUMBER_OF_COVARIATES):
#         gradientBfromUnscaledLinearPart[:, i] += singleU[i] 
#         adjacentNodes = allAdjacentNodes[i]
#         for j in adjacentNodes:
#             gradientBfromUnscaledLinearPart[:, i] += edgesU[i,j]
#     
#     
#     gradientB += rho * gradientBfromReg + gradientBfromUnscaledLinearPart
#     regRuntime = (time.time() - start_time_grad) 
#     
#     return gradientB, gradientBeta, regRuntime

# def optimizeLBFGSNew_unscaled(dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B0, beta0, allAdjacentNodes, MAX_ITERATIONS):
#     NUMBER_OF_CLASSES = B0.shape[0]
#     NUMBER_OF_COVARIATES = B0.shape[1]
#     assert(beta0.shape[1] == NUMBER_OF_CLASSES)
# 
#     TIME_MEASURE_COUNT = [0]
#     GRADIENT_TIME_MEASURE_TOTAL = [0.0]
#     FUNCVALUE_TIME_MEASURE_TOTAL = [0.0]
# 
#     VERSION = "SLOW_VERSION"
#     # VERSION = "FAST_VERSION"
#         
#     if VERSION == "SLOW_VERSION":
#         qUpMinus, zAll = None, None
#         columnWeightsForB = None
#     else:
#         start_time_bStepPrecalculations = time.time()
#         qUpMinus, zAll = precalculationOfAllQsFast_unscaled(edgesZ, edgesU, rho, allAdjacentNodes)
#         columnWeightsForB = numpy.zeros(NUMBER_OF_COVARIATES)
#         for i in xrange(NUMBER_OF_COVARIATES):
#             columnWeightsForB[i] = len(allAdjacentNodes[i]) + 1.0
#             
#         # r = precalculationOfR(pairedZ, singleZ, pairedU, singleU)
#         # print "finished bStepPrecalculations in = ", ((time.time() - start_time_bStepPrecalculations) / (60.0))
#         
#     def funcAndGradForLBFG(X):
#         B, beta = convertBack(X)
#         objValue, regRuntimeFuncValue = getObjValueNew_unscaled(qUpMinus, zAll, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION)
#         gradB, gradBeta, regRuntimeGradient = getGradientNew_unscaled(columnWeightsForB, qUpMinus, dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, VERSION) 
#         
#         FUNCVALUE_TIME_MEASURE_TOTAL[0] += regRuntimeFuncValue
#         GRADIENT_TIME_MEASURE_TOTAL[0] += regRuntimeGradient
#         TIME_MEASURE_COUNT[0] += 1
#     
#         grad = convertToOneVec(gradB, gradBeta)
#         # print "-> objValue = ", objValue
#         # print "grad = ", grad
#         # assert(False)
#         
#         return (objValue, grad)
#     
#     
#     def convertBack(X):
#         Xview = X.view()
#         Xview.shape = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES + 1)
#         B = Xview[:,0:NUMBER_OF_COVARIATES]
#         beta = Xview[:,NUMBER_OF_COVARIATES]
#         beta = beta.reshape((1, NUMBER_OF_CLASSES))
#         return B, beta
#     
#     # runGradientCheck(dataFeatures, dataLabels, pairedZ, singleZ, pairedU, singleU, rho, B0, beta0)
#     
#     # print "START B-STEP"
#     # start_time_bStep = time.time() 
#     
#     bestX, objValue, otherInfo = scipy.optimize.fmin_l_bfgs_b(func = funcAndGradForLBFG, x0 = convertToOneVec(B0, beta0), maxiter = MAX_ITERATIONS)
#     bestB, bestBeta = convertBack(bestX)
#     
#     # print "b-step runtime (sec)" , (time.time() - start_time_bStep) 
#     
#     averageFuncValueTime = FUNCVALUE_TIME_MEASURE_TOTAL[0] / float(TIME_MEASURE_COUNT[0])
#     averageGradientTime =  GRADIENT_TIME_MEASURE_TOTAL[0] / float(TIME_MEASURE_COUNT[0])
#     totalTime = FUNCVALUE_TIME_MEASURE_TOTAL[0] + GRADIENT_TIME_MEASURE_TOTAL[0]
#     # print "number of gradient evaluations = ", TIME_MEASURE_COUNT[0]
#     # print "total time regularizer part in gradient + funcValue (sec) = ", totalTime
#     # print "average time for regularizer part in funcValue (sec) = ", averageFuncValueTime
#     # print "average time for regularizer part in gradient (sec) = ", averageGradientTime
#     # assert(False)
#     
#     return bestB, bestBeta


def test():
     
    # pairedZ = numpy.zeros(shape = (3,5,2))
    # print "pairedZ"
    # print pairedZ[1,4]
    testVec = numpy.asarray([1,2,3])
    testVecAsMat = numpy.outer(testVec, numpy.ones(5))
    print("testVecAsMat = ")
    print(testVecAsMat)
 
# test()
