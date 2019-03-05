
import numpy

# reading checked
def updateU(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
    
    primalResidual = 0.0
    
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i] 
        for j in adjacentNodes:
            residual = edgesZ[i,j] - B[:,i]
            primalResidual += numpy.sum(numpy.square(residual))
            edgesU[i,j] += residual
    
    
    for i in range(NUMBER_OF_COVARIATES):
        residual = singleZ[i] - B[:,i]
        primalResidual += numpy.sum(numpy.square(residual))
        singleU[i] += residual
        
    return numpy.sqrt(primalResidual)


def updateU_fast(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
    
    primalResidual = 0.0
    
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i]
        if len(adjacentNodes) > 0:
            # print adjacentNodes
            # print "B[:,i] = "
            # print B[:,i]
            # print "all adjacent nodes = "
            allResidualsForThisNode = edgesZ[i,adjacentNodes] - B[:,i]
            primalResidual += numpy.sum(numpy.square(allResidualsForThisNode))
            edgesU[i,adjacentNodes] += allResidualsForThisNode
        
    
    
    for i in range(NUMBER_OF_COVARIATES):
        residual = singleZ[i] - B[:,i]
        primalResidual += numpy.sum(numpy.square(residual))
        singleU[i] += residual
        
    return numpy.sqrt(primalResidual)

# reading checked
def updateU_forAcceleration(edgesUhat, singleUhat, edgesZ, singleZ, B, allAdjacentNodes):
    NUMBER_OF_COVARIATES = B.shape[1]
    
    edgesU = numpy.copy(edgesUhat)
    singleU = numpy.copy(singleUhat)
    
    primalResidualSQR = 0.0
    
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = allAdjacentNodes[i] 
        for j in adjacentNodes:
            residual = edgesZ[i,j] - B[:,i]
            primalResidualSQR += numpy.sum(numpy.square(residual))
            edgesU[i,j] += residual
    
    
    for i in range(NUMBER_OF_COVARIATES):
        residual = singleZ[i] - B[:,i]
        primalResidualSQR += numpy.sum(numpy.square(residual))
        singleU[i] += residual
        
    return edgesU, singleU, primalResidualSQR
