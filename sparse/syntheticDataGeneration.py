
import numpy
import helper

def getSyntheticDataProperties(testSetting):
    
    if testSetting == "small":
        NUMBER_OF_CLASSES = 4
        NUMBER_OF_LATENT_CLUSTERS = 10
        NUMBER_OF_COVARIATES_PER_CLUSTER = 4
        IRRELEVANT_CLUSTERS = 0 # 2
        CONTRADICTING_CLUSTERS = 0
    elif testSetting == "smallContra":
        NUMBER_OF_CLASSES = 4
        NUMBER_OF_LATENT_CLUSTERS = 10
        NUMBER_OF_COVARIATES_PER_CLUSTER = 4
        IRRELEVANT_CLUSTERS = 0 # 2
        CONTRADICTING_CLUSTERS = 2
    elif testSetting == "large":
        NUMBER_OF_CLASSES = 4
        NUMBER_OF_LATENT_CLUSTERS = 10
        NUMBER_OF_COVARIATES_PER_CLUSTER = 20 
        IRRELEVANT_CLUSTERS = 0 # 2
        CONTRADICTING_CLUSTERS = 0
    elif testSetting == "largeContra":
        NUMBER_OF_CLASSES = 4
        NUMBER_OF_LATENT_CLUSTERS = 10
        NUMBER_OF_COVARIATES_PER_CLUSTER = 20 
        IRRELEVANT_CLUSTERS = 0 # 2
        CONTRADICTING_CLUSTERS = 2
    elif testSetting == "hugeContra":
        # used only for runtime test
        NUMBER_OF_CLASSES = 4
        NUMBER_OF_LATENT_CLUSTERS = 10
        NUMBER_OF_COVARIATES_PER_CLUSTER = 100
        IRRELEVANT_CLUSTERS = 0
        CONTRADICTING_CLUSTERS = 2
    else:
        assert(False)
        
    return NUMBER_OF_CLASSES, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS

def sampleDataFromMultivariateNormal(classMeanWeights, covariateSims, NUMBER_OF_CLASSES, NUMBER_OF_SAMPLES_PER_CLASS):
    assert(classMeanWeights.shape[0] == NUMBER_OF_CLASSES)
    
    numberOfCovariates = classMeanWeights.shape[1]
    
    dataFeatures = numpy.zeros((NUMBER_OF_CLASSES * NUMBER_OF_SAMPLES_PER_CLASS, numberOfCovariates))
    dataLabels = numpy.zeros(NUMBER_OF_CLASSES * NUMBER_OF_SAMPLES_PER_CLASS)
    
    covarianceMatrix = numpy.copy(covariateSims)
    eigvals = numpy.linalg.eigvalsh(covarianceMatrix)
    assert(numpy.all(eigvals > 0.0)) # must be a positive definite matrix, otherwise we cannot use it as a covariance matrix 
    
    for classId in range(NUMBER_OF_CLASSES):
        samplesForClass = numpy.random.multivariate_normal(mean = classMeanWeights[classId,:], cov = covarianceMatrix, size = NUMBER_OF_SAMPLES_PER_CLASS)
        for i in range(NUMBER_OF_SAMPLES_PER_CLASS):
            dataFeatures[classId * NUMBER_OF_SAMPLES_PER_CLASS + i, :] = samplesForClass[i, :]
            dataLabels[classId * NUMBER_OF_SAMPLES_PER_CLASS + i] = classId
    
    dataLabels = dataLabels.astype(int)
    
    return dataFeatures, dataLabels
    

def createContradictingClusters(hiddenClusterIds, classMeanWeights, NUMBER_OF_COVARIATES_PER_CLUSTER, CONTRADICTING_CLUSTERS):
    
    nrDisruptedCovs = int(NUMBER_OF_COVARIATES_PER_CLUSTER / 2) 
    
    MAX_CLUSTER_ID = numpy.max(hiddenClusterIds)
    
    newHiddenClusterIds = numpy.copy(hiddenClusterIds)
    newClassMeanWeights = numpy.copy(classMeanWeights)
    
    # disrupt the first 'CONTRADICTING_CLUSTERS'-clusters
    for k in range(CONTRADICTING_CLUSTERS):
        nextClusterCovId = (k + 1) * NUMBER_OF_COVARIATES_PER_CLUSTER
        for i in range(nrDisruptedCovs):
            disruptedCovId = k * NUMBER_OF_COVARIATES_PER_CLUSTER + i
            newClassMeanWeights[:,disruptedCovId] = classMeanWeights[:, nextClusterCovId]
            newHiddenClusterIds[disruptedCovId] = MAX_CLUSTER_ID + 1 + k
    
    return newHiddenClusterIds, newClassMeanWeights

def createFixedWeightsMatrix(NUMBER_OF_CLASSES, numberOfRelevantCovariates, numberOfRelevantClusters, hiddenClusterIds, CLASS_STRENGTH, irrelevantCovariates):
    assert(numberOfRelevantCovariates + irrelevantCovariates == hiddenClusterIds.shape[0])
    
    B = numpy.zeros((NUMBER_OF_CLASSES, numberOfRelevantCovariates + irrelevantCovariates))
    
    classWeightIds = numpy.zeros(numberOfRelevantClusters, numpy.int16)
    for k in range(numberOfRelevantClusters):
        classWeightIds[k] = int(k % NUMBER_OF_CLASSES)
    
    for i in range(numberOfRelevantCovariates):
        classId = classWeightIds[hiddenClusterIds[i]]
        B[classId, i] = CLASS_STRENGTH
    
    return B


def createCovariateSimMatrixFixedInOut(hiddenClusterIdsOrig, IN_SIM, OUT_SIM, NUMBER_OF_COVARIATES_PER_CLUSTER, CONTRADICTING_CLUSTERS):
    assert(NUMBER_OF_COVARIATES_PER_CLUSTER % 2 == 0)
    
    hiddenClusterIdsForSim = numpy.copy(hiddenClusterIdsOrig)
    hiddenClusterIds = numpy.copy(hiddenClusterIdsOrig)
    
    HALF_SIZE = int(NUMBER_OF_COVARIATES_PER_CLUSTER / 2)
    highestLabel = numpy.max(hiddenClusterIdsOrig) + 1
    for contraId in range(CONTRADICTING_CLUSTERS):
        startId = HALF_SIZE + contraId * NUMBER_OF_COVARIATES_PER_CLUSTER
        for i in range(NUMBER_OF_COVARIATES_PER_CLUSTER):
            hiddenClusterIdsForSim[startId + i] = highestLabel + contraId
            
        for i in range(HALF_SIZE):
            hiddenClusterIds[startId + i] = highestLabel + contraId * 2
            hiddenClusterIds[startId + HALF_SIZE + i] = highestLabel + contraId * 2 + 1
            
    # print "hiddenClusterIdsForSim = "
    # print hiddenClusterIdsForSim
    # print "hiddenClusterIds = "
    # print hiddenClusterIds
        
    NUMBER_OF_COVARIATES = hiddenClusterIdsForSim.shape[0]
    covariateSims = numpy.zeros((NUMBER_OF_COVARIATES, NUMBER_OF_COVARIATES))
    
    for i1 in range(NUMBER_OF_COVARIATES):
        for i2 in range(i1 + 1,NUMBER_OF_COVARIATES):
            if hiddenClusterIdsForSim[i1] == hiddenClusterIdsForSim[i2]:
                covariateSims[i1,i2] = IN_SIM
            else:
                covariateSims[i1,i2] = OUT_SIM
                
            covariateSims[i2,i1] = covariateSims[i1,i2]
    
    # set diagonal to maxVal
    for i in range(NUMBER_OF_COVARIATES):
        covariateSims[i,i] = 1.0
    
    return covariateSims, hiddenClusterIds

# CONTRADICTING_CLUSTERS specifies the number of times the clustering implied by the prior similarity contradicts with the class label information.
def generateDataMultivariateNormal(NUMBER_OF_CLASSES, NUMBER_OF_SAMPLES_PER_CLASS, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS, TOTAL_NUMBER_OF_FOLDS):
    assert(NUMBER_OF_COVARIATES_PER_CLUSTER >= 2)
    
    numberOfRelevantClusters = NUMBER_OF_LATENT_CLUSTERS - IRRELEVANT_CLUSTERS 
    assert(numberOfRelevantClusters >= NUMBER_OF_CLASSES)
    assert(numberOfRelevantClusters >= CONTRADICTING_CLUSTERS * 2)
    assert(CONTRADICTING_CLUSTERS <= NUMBER_OF_CLASSES)
        
    NUMBER_OF_COVARIATES = NUMBER_OF_LATENT_CLUSTERS * NUMBER_OF_COVARIATES_PER_CLUSTER
    
    hiddenClusterIds = numpy.zeros(NUMBER_OF_COVARIATES, numpy.int16)
    
    for k in range(NUMBER_OF_LATENT_CLUSTERS):
        for i in range(NUMBER_OF_COVARIATES_PER_CLUSTER):
            covariateId = k * NUMBER_OF_COVARIATES_PER_CLUSTER + i
            hiddenClusterIds[covariateId] = k
    
    
    CLASS_STRENGTH = 5.0
    numberOfRelevantCovariates = numberOfRelevantClusters * NUMBER_OF_COVARIATES_PER_CLUSTER
    irrelevantCovariates = IRRELEVANT_CLUSTERS * NUMBER_OF_COVARIATES_PER_CLUSTER
    classMeanWeights = createFixedWeightsMatrix(NUMBER_OF_CLASSES, numberOfRelevantCovariates, numberOfRelevantClusters, hiddenClusterIds, CLASS_STRENGTH, irrelevantCovariates)
    
    relevantCovariates = numpy.zeros(NUMBER_OF_COVARIATES, numpy.int16)
    relevantCovariates[0:numberOfRelevantCovariates] = 1
    
    IN_SIM = 0.9
    OUT_SIM = 0.0        
    covariateSims, hiddenClusterIds = createCovariateSimMatrixFixedInOut(hiddenClusterIds, IN_SIM, OUT_SIM, NUMBER_OF_COVARIATES_PER_CLUSTER, CONTRADICTING_CLUSTERS)
    
    if NUMBER_OF_COVARIATES <= 50:
        print("final covariateSims = ")
        helper.showMatrix(covariateSims)
    
    print("finished creating covariate similarity matrix")
    
    # ensure that all irrelevant clusters have the same clusterId
    irrelevantClusterId = numberOfRelevantClusters
    for k in range(IRRELEVANT_CLUSTERS):
        for i in range(NUMBER_OF_COVARIATES_PER_CLUSTER):
            covariateId = (numberOfRelevantClusters + k) * NUMBER_OF_COVARIATES_PER_CLUSTER + i
            hiddenClusterIds[covariateId] = irrelevantClusterId
    
    if NUMBER_OF_COVARIATES <= 100:
        print("hiddenClusterIds: ")
        helper.showVecInt(hiddenClusterIds)
        print("final classMeanWeights = ")
        helper.showMatrix(classMeanWeights)
        
    print("number of cluster = ", len(set(hiddenClusterIds)))
    assert(classMeanWeights.shape[1] == NUMBER_OF_COVARIATES)
    print("NUMBER_OF_COVARIATES = ", NUMBER_OF_COVARIATES)
    
    dataFeature_allFolds = []
    dataLabels_allFolds = []
    for foldId in range(TOTAL_NUMBER_OF_FOLDS):
        dataFeatures, dataLabels =  sampleDataFromMultivariateNormal(classMeanWeights, covariateSims, NUMBER_OF_CLASSES, NUMBER_OF_SAMPLES_PER_CLASS)
        dataFeature_allFolds.append(dataFeatures)
        dataLabels_allFolds.append(dataLabels)
        
    return covariateSims, dataFeature_allFolds, dataLabels_allFolds, hiddenClusterIds, relevantCovariates


def generateData(testSetting, NUMBER_OF_SAMPLES_PER_CLASS, TOTAL_NUMBER_OF_FOLDS):
    
    NUMBER_OF_CLASSES, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS = getSyntheticDataProperties(testSetting) 

    NUMBER_OF_COVARIATES = NUMBER_OF_LATENT_CLUSTERS * NUMBER_OF_COVARIATES_PER_CLUSTER
    numpy.random.seed(4324239)
    covariateSims, dataFeature_allFolds, dataLabels_allFolds, trueClusterIds, trueRelevantCovariates = generateDataMultivariateNormal(NUMBER_OF_CLASSES, NUMBER_OF_SAMPLES_PER_CLASS, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS, TOTAL_NUMBER_OF_FOLDS)
    print("p = ", len(trueClusterIds))
    print("n = ", dataFeature_allFolds[0].shape[0])
    # assert(False)
    return covariateSims, dataFeature_allFolds, dataLabels_allFolds, trueClusterIds, trueRelevantCovariates, NUMBER_OF_CLASSES, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS

