import baselineMultiLogReg
import numpy
import bStep
import helper

# checked
# returns S(B) (see paper) which is a strictly negative definite matrix
# sigma specifies the standard deviation from the gaussian prior on B
def getFullHessian(dataFeatures, dataLabels, B, beta, sigma):
    assert(sigma > 0.0)
    NUMBER_OF_CLASSES = B.shape[0]
    NUMBER_OF_COVARIATES = B.shape[1]
  
    # rows = samples, columns = class-labels
    probsEachSample = bStep.getAllProbs(dataFeatures, B, beta)
    
    fullHessian = numpy.zeros((NUMBER_OF_CLASSES * NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES * NUMBER_OF_COVARIATES))
    for i in range(NUMBER_OF_CLASSES):
        for j in range(i,NUMBER_OF_CLASSES, 1):
            hessianBlock = numpy.zeros((NUMBER_OF_COVARIATES,NUMBER_OF_COVARIATES))
            for sampleId in range(dataFeatures.shape[0]):
                if i == j:
                    p = probsEachSample[sampleId, i] 
                    fac = (p - 1.0) * p
                else:
                    fac = probsEachSample[sampleId, i] * probsEachSample[sampleId, j]
                xxT = numpy.asmatrix(dataFeatures[sampleId]).transpose() * numpy.asmatrix(dataFeatures[sampleId]) 
                hessianBlock += fac * xxT
            fullHessian[(i * NUMBER_OF_COVARIATES) : ((i+1) * NUMBER_OF_COVARIATES), (j * NUMBER_OF_COVARIATES) : ((j+1) * NUMBER_OF_COVARIATES)] = hessianBlock
            fullHessian[(j * NUMBER_OF_COVARIATES) : ((j+1) * NUMBER_OF_COVARIATES), (i * NUMBER_OF_COVARIATES) : ((i+1) * NUMBER_OF_COVARIATES)] = hessianBlock
    
    fullHessian -= (1.0 / (sigma ** 2)) * numpy.eye(NUMBER_OF_CLASSES * NUMBER_OF_COVARIATES)
    
    # print "FINISHED CALCULATION OF FULL HESSIAN"
    return fullHessian


def getDiagHessian(dataFeatures, dataLabels, B, beta, sigma):
    assert(sigma > 0.0)
    NUMBER_OF_CLASSES = B.shape[0]
    NUMBER_OF_COVARIATES = B.shape[1]
  
    # rows = samples, columns = class-labels
    probsEachSample = bStep.getAllProbs(dataFeatures, B, beta)
    
    diagHessian = numpy.zeros(NUMBER_OF_CLASSES * NUMBER_OF_COVARIATES)
    for i in range(NUMBER_OF_CLASSES):
        hessianBlock = numpy.zeros(NUMBER_OF_COVARIATES)
        for sampleId in range(dataFeatures.shape[0]):
            p = probsEachSample[sampleId, i] 
            fac = (p - 1.0) * p
            xSquare = numpy.square(dataFeatures[sampleId])
            assert(fac <= 0) 
            hessianBlock += fac * xSquare
        diagHessian[(i * NUMBER_OF_COVARIATES) : ((i+1) * NUMBER_OF_COVARIATES)] = hessianBlock
    
    diagHessian -= (1.0 / (sigma ** 2))
    return diagHessian


def evalClusteringNew(origDataFeatures, dataLabels, sortedClusters, fullCov, bestLambdaParamFullModel):
    
    # print "BEFORE PROJECTION:"
    # print "NUMBER_OF_COVARIATES = ", origDataFeatures.shape[1]
    
    dataFeatures = helper.projectData(origDataFeatures, sortedClusters)
    
    NUMBER_OF_COVARIATES = dataFeatures.shape[1]
    NUMBER_OF_CLASSES = numpy.max(dataLabels) + 1
    assert(dataFeatures.shape[0] == dataLabels.shape[0])
        
    # print "AFTER PROJECTION:"
    # print "NUMBER_OF_COVARIATES = ", NUMBER_OF_COVARIATES
    # print "NUMBER_OF_CLASSES = ", NUMBER_OF_CLASSES
    
    MAX_LBFGS_ITERATIONS = 15000 # default value
    B0 = numpy.random.normal(size = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES))
    beta0 = numpy.random.normal(size = (1,NUMBER_OF_CLASSES))
    
    
    sigma = numpy.sqrt(1.0 / (2.0 * bestLambdaParamFullModel))
    bestLambdaParam = bestLambdaParamFullModel
    
    trainCVlogProb, _ = baselineMultiLogReg.estimateHeldOutPerformance(dataFeatures, dataLabels, "logProb", bestLambdaParam)
    trainCVacc, _ = baselineMultiLogReg.estimateHeldOutPerformance(dataFeatures, dataLabels, "accuracy", bestLambdaParam)
    
    
    if NUMBER_OF_COVARIATES == 0:
        print("NO COVARIATES AFTER PROJECTION")
        return float("-inf"), B0, sigma, float("-inf"), float("-inf")
    
    B_MAP, beta_MAP, negLogLiklihood = baselineMultiLogReg.optimizeLBFGS(dataFeatures, dataLabels, B0, beta0, bestLambdaParam, MAX_LBFGS_ITERATIONS)
    
    # add the normalization term from the normal prior
    logJointProb = - negLogLiklihood - 0.5 * (NUMBER_OF_COVARIATES * NUMBER_OF_CLASSES) * numpy.log(2.0 * numpy.pi * sigma * sigma)
    
    if fullCov:
        fullHessian = getFullHessian(dataFeatures, dataLabels, B_MAP, beta_MAP, sigma)
        s, logDetS = numpy.linalg.slogdet(- fullHessian)
        assert(s > 0)
    else:
        diagHessian = getDiagHessian(dataFeatures, dataLabels, B_MAP, beta_MAP, sigma)
        logDetS = numpy.sum(numpy.log(-diagHessian))
    
    logMarginalLikelihood = logJointProb 
    logMarginalLikelihood += 0.5 * (NUMBER_OF_COVARIATES * NUMBER_OF_CLASSES) * numpy.log(2.0 * numpy.pi)
    logMarginalLikelihood -= 0.5 * logDetS
        
    return logMarginalLikelihood, B_MAP, sigma, trainCVlogProb, trainCVacc 


    
                   
    