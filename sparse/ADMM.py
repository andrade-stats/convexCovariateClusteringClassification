
import numpy
import bStep
import zStep
import uStep
import helper
import time
import sklearn.metrics


def parallelADMM(dataFeatures, dataLabels, covariateSims, origNu, origGamma, optParams, warmStartAuxilliaryVars, paramId):
    assert(covariateSims.shape[0] ==  covariateSims.shape[1])
    assert(numpy.min(dataLabels) == 0)
    
    # scale nu and gamma to account for different training data sizes
    N = float(dataLabels.shape[0])
    nu = N * origNu
    gamma = N * origGamma
    
    NUMBER_OF_COVARIATES = covariateSims.shape[0]
    NUMBER_OF_CLASSES = numpy.max(dataLabels) + 1
    
    print("NUMBER_OF_COVARIATES = ", NUMBER_OF_COVARIATES)
    print("NUMBER_OF_CLASSES = ", NUMBER_OF_CLASSES)
    
    rho = optParams["INITIAL_RHO"]
    rhoMultiplier = optParams["RHO_MULTIPLIER"]
    assert(rhoMultiplier == 1.0)
    
    MAX_LBFGS_ITERATIONS = optParams["MAX_LBFGS_ITERATIONS"]
    ADMM_MAX_ITERATIONS = optParams["ADMM_MAX_ITERATIONS"]
    
    print("covariateSims (part of it) = ")
    helper.showMatrix(covariateSims[0:10,0:10])
   
    totalNumberOfNeighbours = 0
    
    # fully connected graph
    NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES = 0
    allAdjacentNodes = []
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = []
        for j in range(NUMBER_OF_COVARIATES):
            if i != j and covariateSims[i,j] > 0.0:
                adjacentNodes.append(j)
                NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES += 1
        
        totalNumberOfNeighbours += len(adjacentNodes)
        allAdjacentNodes.append(numpy.asarray(adjacentNodes)) 
    
    print("average number of neighbours = ", (float(totalNumberOfNeighbours) / float(NUMBER_OF_COVARIATES)))
    print("total number of edges in graph = ", NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES)
    
    NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES += NUMBER_OF_COVARIATES
    print("NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES = ", NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES)
    
    
    B = numpy.random.normal(size = (NUMBER_OF_CLASSES, NUMBER_OF_COVARIATES))
    beta = numpy.random.normal(size = (1,NUMBER_OF_CLASSES))
    
    if warmStartAuxilliaryVars is None:
        edgesZ = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
        edgesU = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
        singleZ = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
        singleU = numpy.zeros(shape = (NUMBER_OF_COVARIATES, NUMBER_OF_CLASSES))
    else:
        edgesZ, edgesU, singleZ, singleU = warmStartAuxilliaryVars
        
    EPSILON = optParams["EPSILON"]
    
    CONVERGENCE_THRESHOLD = numpy.sqrt(NUMBER_OF_ACTIVE_AUXILIARY_VARIABLES * NUMBER_OF_CLASSES) * EPSILON
    print("CONVERGENCE_THRESHOLD = ", CONVERGENCE_THRESHOLD)
    
    start_time_ADMM_run = time.time()
  
    # nrRelevantClusters, clusterIds, relevance = None, None, None 
    clusteringInfoFullConnected, clusteringInfoPartialConnected = None, None
  
    converged = None
    
    for it in range(ADMM_MAX_ITERATIONS):
        start_time_one_it = time.time()
        print("PARAMID = ", paramId)
        print("iteration ", it)
        print("rho = ", rho)
        
        # WARNING: calculation of getObjValueFullModel is slow !!
        # currentObjValue = getObjValueFullModel(dataFeatures, dataLabels, covariateSims, B, beta, nu, gamma)
        # print "*** currentObjValue = ", currentObjValue
    
        start_time_bStep = time.time()
        B, beta = bStep.optimizeLBFGSNew(dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, MAX_LBFGS_ITERATIONS)
        print("finished bStep in = ", ((time.time() - start_time_bStep) / (60.0)))
        
        start_time_zStep = time.time()
        dualResidual = zStep.updateZ_fast(edgesZ, singleZ, edgesU, singleU, B, rho, covariateSims, nu, gamma, allAdjacentNodes)
        print("finished zStep in = ", ((time.time() - start_time_zStep) / (60.0)))
        
        # assert(False)
        start_time_uStep = time.time()
        primalResidual = uStep.updateU_fast(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes)
        print("finished uStep in = ", ((time.time() - start_time_uStep) / (60.0)))
          
              
        print("primal residual = ", primalResidual)
        print("dual residual = ", dualResidual)
        print("combined = ", (primalResidual + dualResidual))
        
        if (primalResidual < CONVERGENCE_THRESHOLD) and (dualResidual < CONVERGENCE_THRESHOLD) and it >= 10:
            print("SUCESSFULLY CONVERGED")
            # run one last iteration and the get clustering from the thresholded Z variables
            B, beta = bStep.optimizeLBFGSNew(dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B, beta, allAdjacentNodes, MAX_LBFGS_ITERATIONS)
            clusteringInfoPartialConnected = zStep.getClusteringAndSelectedFeatures(edgesU, singleU, B, rho, covariateSims, nu, gamma, allAdjacentNodes)
            converged = True
            break
        else:
            if rho < 0.1:
                rho = rho * rhoMultiplier
        
        # print "total dual residual = ", getDualResidual(allZPrevious, allZ, NUMBER_OF_COVARIATES)
        # allZPrevious = backupAllZ(allZ, NUMBER_OF_COVARIATES)
        # print "finished iteration in = ", ((time.time() - start_time_one_it) / (60.0))
        # assert(False)
    else:
        print("!! WARNING: DID NOT CONVERGE !!")
        print("rho = ", rho)
        print("EPSILON = ", EPSILON)
        print("origNu = ", origNu)
        print("origGamma = ", origGamma)
        clusteringInfoPartialConnected = zStep.getClusteringAndSelectedFeatures(edgesU, singleU, B, rho, covariateSims, nu, gamma, allAdjacentNodes)
        converged = False
        
    assert(converged is not None)
    currentObjValue = getObjValueFullModel(dataFeatures, dataLabels, covariateSims, B, beta, nu, gamma)
    print("final objective value = ", currentObjValue)

    duration_ADMM_run = (time.time() - start_time_ADMM_run) 
    print("-----------------")
    print("Finished in (min) = ", round(duration_ADMM_run / (60.0),3))
    print("-----------------")
    
    return B, beta, clusteringInfoPartialConnected, converged, (edgesZ, edgesU, singleZ, singleU)



# # checked
def getObjValueFullModel(dataFeatures, dataLabels, S, B, beta, nu, gamma):
     
    start_time_getObjValueFullModel = time.time()
     
    NUMBER_OF_COVARIATES = B.shape[1]
    totalNegLogProb = bStep.getTotalNegLogProb(dataFeatures, dataLabels, B, beta)
     
    totalRegClustering = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        for j in range(i+1,NUMBER_OF_COVARIATES,1):
            assert(S[i,j] == S[j,i])
            totalRegClustering += S[i,j] * helper.l2Norm(B[:,i] - B[:,j])
     
    totalRegSparsity = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        totalRegSparsity += helper.l2Norm(B[:,i])
     
    # print "totalNegLogProb = ", totalNegLogProb
    # print "totalRegClustering = ", totalRegClustering
    # print "totalRegSparsity = ", totalRegSparsity
     
    print("finished objective calculation in = ", ((time.time() - start_time_getObjValueFullModel) / (60.0)))
     
    return totalNegLogProb + nu * totalRegClustering + gamma * totalRegSparsity


def evaluate(evalData, trueLabels, B, beta):
    predictedLabels = bStep.predictLabels(evalData, B, beta)
    microScore = sklearn.metrics.f1_score(y_true = trueLabels, y_pred = predictedLabels, average='micro')
    macroScore = sklearn.metrics.f1_score(y_true = trueLabels, y_pred = predictedLabels, average='macro')
    accuracy = sklearn.metrics.accuracy_score(y_true = trueLabels, y_pred = predictedLabels)
    return microScore, macroScore, accuracy

