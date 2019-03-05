
import numpy
import helper
import scipy.sparse

# import cvxpy


# solves for argmin_{z1, z2} s * ||z1 - z2|| + (rho/2) * ||z1 - m1|| + (rho/2) * ||z2 - m2||
def optimizeCVX(rho, s, m1, m2):
   
    d = m1.shape[0]
    z1 = cvxpy.Variable(d) # z_{i->j}
    z2 = cvxpy.Variable(d) # z_{j->i}
        
    prob = cvxpy.Problem(cvxpy.Minimize(s * cvxpy.norm(z1 - z2)
                        + (rho/2.0) * (cvxpy.sum_squares(z1 - m1) + cvxpy.sum_squares(z2 - m2)) ))
    prob.solve(solver = cvxpy.ECOS)
    
    # print "CVX result: "
    # print "status = ", prob.status
    # print "Optimal value = ", prob.value

    # print numpy.asarray(z1.value)[:,0]
    
    return numpy.asarray(z1.value)[:,0], numpy.asarray(z2.value)[:,0]


# checked
# solves for argmin_{z1, z2} s * ||z1 - z2|| + (rho/2) * ||z1 - m1||^2 + (rho/2) * ||z2 - m2||^2
# s = v * S_{ij}
# m1 = b_i - u_{i->j}
# m2 = b_j - u_{j->i}
# 
# for unscaled version use
# m1 = b_i + (1/rho)* u_{i->j}
# m2 = b_j + (1/rho)* u_{j->i}
def optimizeWithAnalyticSolution(rho, s, m1, m2):
    
    mag = helper.l2Norm(m1 - m2)
    if mag <= 0.0:
        theta = 0.5
    else:
        fac = s / (rho * mag)
        theta = numpy.max([1.0 - fac, 0.5])
        
    z1 = theta * m1 + (1.0 - theta) * m2
    z2 = (1.0 - theta) * m1 + theta * m2
    return z1, z2

# based on "optimizeWithAnalyticSolution" checks whether theta is 0.5 or not
def edgeWeightsEqual(rho, s, m1, m2):
    
    mag = helper.l2Norm(m1 - m2)
    if mag <= 0.0:
        return True
    else:
        fac = s / (rho * mag)
        return ((1.0 - fac) <= 0.5)
    
    
# reading checked
def updateZ(edgesZ, singleZ, edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
        
    dualResidual = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i] 
        for j in adjacentNodes:
            if j > i:
                s = nu * S[i,j]
                m1 = B[:,i] - edgesU[i,j]
                m2 = B[:,j] - edgesU[j,i]
                z1, z2 = optimizeWithAnalyticSolution(rho, s, m1, m2)
                dualResidual += numpy.sum(numpy.square(edgesZ[i,j] - z1))
                dualResidual += numpy.sum(numpy.square(edgesZ[j,i] - z2))
                edgesZ[i,j] = z1 # update edge
                edgesZ[j,i] = z2 # update edge
        
    for i in range(NUMBER_OF_COVARIATES):
        fac = gamma / rho
        vec = B[:,i] - singleU[i]
        z = helper.blockThreshold(fac, vec)
        dualResidual += numpy.sum(numpy.square(singleZ[i] - z))
        singleZ[i] = z # update single z 
    
    return numpy.sqrt(dualResidual)


def isConnectedToAll(allAdjacentNodes, nodesInCluster, queryNodeId):
    fullyConnected = True
    for i in nodesInCluster:
        if (i not in allAdjacentNodes[queryNodeId]):
            fullyConnected = False
            break
    
    return fullyConnected
            
# is only an heuristic ! get the maximal cliques is NP-hard problem
def getFullyConnectedClusters(allAdjacentNodes, NUMBER_OF_COVARIATES):
    
    allClusterIds = numpy.zeros(NUMBER_OF_COVARIATES, dtype=numpy.int32)
    
    coveredNodes = set()
    clusterId = 1
    for i in range(NUMBER_OF_COVARIATES):
        if i not in coveredNodes:
            nodesInCluster = set()
            nodesInCluster.add(i)
            for j in allAdjacentNodes[i]:
                if isConnectedToAll(allAdjacentNodes, nodesInCluster, j):
                    nodesInCluster.add(j)
             
            # print "nodesInCluster = ", nodesInCluster 
            ids = numpy.asarray(list(nodesInCluster), dtype=numpy.int32)
            assert(ids.shape[0] == len(nodesInCluster))
            # print "ids = ", ids
            # print "allClusterIds (before) = "
            # print allClusterIds
            allClusterIds[ids] = clusterId
            # print "allClusterIds (after) = "
            # print allClusterIds
            
            coveredNodes.update(nodesInCluster)
            clusterId += 1
    
    # print "NUMBER_OF_COVARIATES = ", NUMBER_OF_COVARIATES
    # print "len(coveredNodes) = ", len(coveredNodes)
    # print "lenght = ", len(allClusterIds[allClusterIds == 0])
    # print numpy.where(allClusterIds == 0)
    assert(len(allClusterIds[allClusterIds == 0]) == 0)
    return allClusterIds
    

def getFinalClusterIdsAndRelevance(origClusterIds, origRelevance):
    NUMBER_OF_COVARIATES = origClusterIds.shape[0]
    clusterIds = numpy.copy(origClusterIds)
    relevance = numpy.copy(origRelevance)
    assert(clusterIds.shape[0] == relevance.shape[0])
    
    # make sure that all irrelevant features are in the same cluster
    IRRELEVANT_CLUSTER_ID = numpy.max(clusterIds) + 1
    for i in range(NUMBER_OF_COVARIATES):
        if relevance[i] == 0:
            currentClusterId = clusterIds[i]
            # mark all features in this cluster as irrelevant
            irrelevantIds = numpy.where(clusterIds == currentClusterId)[0]
            assert(irrelevantIds.shape[0] >= 1)
            clusterIds[irrelevantIds] = IRRELEVANT_CLUSTER_ID
            relevance[irrelevantIds] = 0
    
    newNrIrrelevantFeatures = numpy.count_nonzero(clusterIds == IRRELEVANT_CLUSTER_ID)
    if newNrIrrelevantFeatures != numpy.count_nonzero(origRelevance == 0):
        print("WARNING: METHOD CONVERGED NOT PROPERLY - THE IRRELEVANT FEATURES ARE UNCLEAR")
        print("Number of irrelevant features with thresholding only = ", numpy.count_nonzero(origRelevance == 0))
        print("Number of irrelevant features with all irrelevant when in same cluster = ", newNrIrrelevantFeatures)
        print("Number of irrelevant feature (debug) = ", numpy.count_nonzero(relevance == 0))
    else:
        print("SUCCESSFULLY DETERMINED IRRELEVANT FEATURES")

    assert(newNrIrrelevantFeatures == numpy.count_nonzero(relevance == 0))

    if newNrIrrelevantFeatures > 0:
        nrRelevantClusters = len(set(clusterIds)) - 1
    else:
        nrRelevantClusters = len(set(clusterIds)) 
        
    print("number of relevant clusters = ", nrRelevantClusters)
    print("number of irrelevant features = ", newNrIrrelevantFeatures)
    
    return clusterIds, relevance


# reading checked
def getClusteringAndSelectedFeatures(edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodesPrior):
    NUMBER_OF_COVARIATES = B.shape[1]
    
    relevance = numpy.ones(NUMBER_OF_COVARIATES, numpy.int32)
    for i in range(NUMBER_OF_COVARIATES):
        fac = gamma / rho
        vec = B[:,i] - singleU[i]
        if helper.isBlockThresholdedToZero(fac, vec):
            relevance[i] = 0
            
    graphMatrix = scipy.sparse.dok_matrix((NUMBER_OF_COVARIATES, NUMBER_OF_COVARIATES), dtype=numpy.int32)
    allAdjacentNodesPosterior = {}
    for i in range(NUMBER_OF_COVARIATES):
        allAdjacentNodesPosterior[i] = set()
    
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodesPrior[i]
        for j in adjacentNodes:
            if j > i:
                s = nu * S[i,j]
                m1 = B[:,i] - edgesU[i,j]
                m2 = B[:,j] - edgesU[j,i]
                if edgeWeightsEqual(rho, s, m1, m2):
                    # there is an edge between node i and node j
                    graphMatrix[i,j] = 1
                    allAdjacentNodesPosterior[i].add(j)
                    allAdjacentNodesPosterior[j].add(i)
    
    # fullyConnectedClusterIds = getFullyConnectedClusters(allAdjacentNodesPosterior, NUMBER_OF_COVARIATES)
    
    nComponents, partiallyConnectedClusterIds = scipy.sparse.csgraph.connected_components(graphMatrix, directed=False)
    partiallyConnectedClusterIds += 1
    assert(nComponents == numpy.max(partiallyConnectedClusterIds))
    
    return getFinalClusterIdsAndRelevance(partiallyConnectedClusterIds, relevance) 
    
def optimizeWithAnalyticSolutionFast(rho, s, m1, m2):
    v = m1 - m2
    mag = numpy.linalg.norm(v, axis=1)
#     print "mag = "
#     print mag
#     print "ref = "
#     print helper.l2Norm(v[0])
#     print helper.l2Norm(v[1])
#     print helper.l2Norm(v[2])
#     assert(False)
    
    fac = s / (rho * mag)
    theta = 1.0 - fac
    theta[theta < 0.5] = 0.5
     
    m1T = m1.transpose()
    m2T = m2.transpose()
    z1 = theta * m1T + (1.0 - theta) * m2T
    z2 = (1.0 - theta) * m1T + theta * m2T
    return z1.transpose(), z2.transpose()


def updateZ_fast(edgesZ, singleZ, edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
        
    dualResidual = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i] 
        if len(adjacentNodes) > 0:
            # assert(False)
            # this should be speed up:
            # filteredAdjacentNodes = [j for j in adjacentNodes if j > i]
            filteredAdjacentNodes = adjacentNodes[adjacentNodes > i]
            # print "filteredAdjacentNodes = ", filteredAdjacentNodes
            # assert(False)
            
            s = nu * S[i,filteredAdjacentNodes]
            m1 = B[:,i] - edgesU[i,filteredAdjacentNodes]
            m2 = (B[:,filteredAdjacentNodes]).transpose() - edgesU[filteredAdjacentNodes,i]
            z1, z2 = optimizeWithAnalyticSolutionFast(rho, s, m1, m2)
            
            dualResidual += numpy.sum(numpy.square(edgesZ[i,filteredAdjacentNodes] - z1))
            dualResidual += numpy.sum(numpy.square(edgesZ[filteredAdjacentNodes,i] - z2))
            edgesZ[i,filteredAdjacentNodes] = z1 # update edge
            edgesZ[filteredAdjacentNodes,i] = z2 # update edge
        
    
    for i in range(NUMBER_OF_COVARIATES):
        fac = gamma / rho
        vec = B[:,i] - singleU[i]
        z = helper.blockThreshold(fac, vec)
        dualResidual += numpy.sum(numpy.square(singleZ[i] - z))
        singleZ[i] = z # update single z 
    
    return numpy.sqrt(dualResidual)


# reading checked
def updateZ_forAcceleration(edgesZhat, singleZhat, edgesUhat, singleUhat, B, rho, S, nu, gamma, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
    
    edgesZ = numpy.zeros_like(edgesZhat) 
    singleZ = numpy.zeros_like(singleZhat)
    
    dualResidualSQR = 0.0
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i] 
        for j in adjacentNodes:
            if j > i:
                s = nu * S[i,j]
                m1 = B[:,i] - edgesUhat[i,j]
                m2 = B[:,j] - edgesUhat[j,i]
                z1, z2 = optimizeWithAnalyticSolution(rho, s, m1, m2)
                dualResidualSQR += numpy.sum(numpy.square(edgesZhat[i,j] - z1))
                dualResidualSQR += numpy.sum(numpy.square(edgesZhat[j,i] - z2))
                edgesZ[i,j] = z1 # update edge
                edgesZ[j,i] = z2 # update edge
        
    for i in range(NUMBER_OF_COVARIATES):
        fac = gamma / rho
        vec = B[:,i] - singleUhat[i]
        z = helper.blockThreshold(fac, vec)
        dualResidualSQR += numpy.sum(numpy.square(singleZhat[i] - z))
        singleZ[i] = z # update single z 
    
    return edgesZ, singleZ, dualResidualSQR

