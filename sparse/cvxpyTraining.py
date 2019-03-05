import cvxpy
import numpy
import time
import scipy
import experiments
import zStep

from multiprocessing import Pool

# should not be used anymore
def matches(vec1, vec2, scale):
    return numpy.sum(numpy.abs(vec1 - vec2)) / float(vec1.shape[0]) < scale * 0.001
      
  
# reading checked
def getClusteringAndRelevanceForCVXPY(B):
    NUMBER_OF_COVARIATES = B.shape[1]
     
    scale = numpy.std(B)
     
    graphMatrix = scipy.sparse.dok_matrix((NUMBER_OF_COVARIATES, NUMBER_OF_COVARIATES), dtype=numpy.int32)
     
    for i in range(NUMBER_OF_COVARIATES):
        refVec = B[:,i]
        for j in range(i + 1, NUMBER_OF_COVARIATES, 1):
            if matches(refVec, B[:,j], scale):
                # there is an edge between node i and node j
                graphMatrix[i,j] = 1
             
    nComponents, clusterIds = scipy.sparse.csgraph.connected_components(graphMatrix, directed=False)
    clusterIds += 1
 
    relevance = numpy.ones(NUMBER_OF_COVARIATES, numpy.int32)
    for i in range(NUMBER_OF_COVARIATES):
        if matches(B[:,i],0, scale):
            relevance[i] = 0
 
 
    nComponents, partiallyConnectedClusterIds = scipy.sparse.csgraph.connected_components(graphMatrix, directed=False)
    partiallyConnectedClusterIds += 1
    assert(nComponents == numpy.max(partiallyConnectedClusterIds))
    
    return zStep.getFinalClusterIdsAndRelevance(partiallyConnectedClusterIds, relevance)
    


# finished compilation in (min) =  13.411

# checked
def proposedMethodTrainedWithCVXPY(BASE_FILENAME, dataFeatures, dataLabels, covariateSims, allNus, allGammas, optParams):
    assert(dataLabels.shape[0] == dataFeatures.shape[0])
        
    start_time_allExperiments = time.time()
       
    n = dataFeatures.shape[0]
    d = dataFeatures.shape[1]
    k = dataLabels.max() + 1
   
    # S = cvxpy.Parameter(shape=(d,d), nonneg=True)
    # S.value = covariateSims
   
    nu = cvxpy.Parameter(nonneg=True)
    # gamma = cvxpy.Parameter(nonneg=True)
        
    B = cvxpy.Variable(shape = (k, d))
    b0 = cvxpy.Variable(shape = (k))
        
    states = []
    for i1 in range(d):
        for i2 in range(i1 + 1,d):
            # print str(i1) + "," + str(i2)
            if covariateSims[i1,i2] > 0.0:
                regularizer = nu * covariateSims[i1,i2] * cvxpy.norm(B[:,i1] - B[:,i2], 2)
                constraints = []
                states.append(cvxpy.Problem(cvxpy.Minimize(regularizer), constraints))
    
    # for i in range(d):
    #     regularizer = gamma * cvxpy.norm(B[:,i], 2)
    #    constraints = []
    #    states.append(cvxpy.Problem(cvxpy.Minimize(regularizer), constraints))
    
    for s in range(n):
        loss = - (B[dataLabels[s],:] * dataFeatures[s,:].T + b0[dataLabels[s]]) + cvxpy.log_sum_exp(B * dataFeatures[s,:].T + b0)
        constraints = []
        states.append(cvxpy.Problem(cvxpy.Minimize(loss), constraints) ) 
    
    # sums problem objectives and concatenates constraints.
    prob = sum(states)
    print("finished compilation in (min) = ", round((time.time() - start_time_allExperiments) / (60.0),3))
    # assert(False)
    
    print("Start running cvx....")
    
    assert(len(allGammas) == 1)
    assert(allGammas[0] == 0.0)
       
    for nuValue in allNus:
        
        nu.value = n * nuValue
        
        try:
            # confirmed that it uses several cores
            if optParams["SOLVER"] == "ECOS":
                prob.solve(solver = cvxpy.ECOS, warm_start=True, verbose=True) # standard options
                # prob.solve(solver = cvxpy.ECOS, max_iters = optParams["ADMM_MAX_ITERATIONS"], abstol = optParams["EPSILON"], reltol = optParams["EPSILON"], feastol=optParams["EPSILON"], warm_start=True, verbose=True)
            elif optParams["SOLVER"] == "SCS":
                prob.solve(solver = cvxpy.SCS, max_iters = optParams["ADMM_MAX_ITERATIONS"], eps = optParams["EPSILON"], warm_start=True, verbose=True)
            else:
                assert(False)
                
            print("status = ", prob.status)
            print("Optimal value = ", prob.value)
            
            if prob.value is None:
                # did not converge 
                print("!!! WARNING: DID NOT CONVERGE !!!")
                finalB = numpy.zeros(shape = (k, d))
                finalb0 = numpy.zeros(shape = (k))
            else:
                finalB = B.value
                finalb0 = b0.value
        
        except (KeyboardInterrupt):
            assert(False)
        except:
            # did not converge 
            print("!!! WARNING: ERROR IN SOLVER AND DID NOT CONVERGE !!!")
            finalB = numpy.zeros(shape = (k, d))
            finalb0 = numpy.zeros(shape = (k))
        
        
        partialConnectedInfo = getClusteringAndRelevanceForCVXPY(finalB)
        experiments.saveStatistics(BASE_FILENAME, optParams, nuValue, 0.0, partialConnectedInfo, finalB, finalb0)
 
    
    duration = (time.time() - start_time_allExperiments)
    
    print("-----------------")
    print("Finished successfully full training in (min) = ", round(duration / (60.0),3))
    print("-----------------")

    return duration
    


