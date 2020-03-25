
import numpy
import scipy.misc
import sklearn.model_selection
import sklearn.metrics

# import tools.prepare_data as prepare
# import tools.io_helper as io_helper
# import tools.load_documents as load_documents
# import tools.math_helpers as math_helpers

PATH_TO_SOURCE_FOLDER = "../"


LAMBDA = 0.1 # default value

TOTAL_NUMBER_OF_FOLDS_FOR_SYNTHETIC_DATA = 10  # default value

DEFAULT_MAX_LBFGS_ITERATIONS = 15000 # default value
DEFAULT_ADMM_MAX_ITERATIONS = 1000 # default value 
DEFAULT_EPSILON = 0.00001 # default value

def getHyperparameters(hyperparametersRange):
    if hyperparametersRange == "coarse":
        allNus =    [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        allGammas = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    elif hyperparametersRange == "joint":
        allNus =    [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.0]
        allGammas = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.0]
    elif hyperparametersRange == "jointFine":
        allNus =    [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001, 0.000000001, 0.0000000001, 0.0]
        allGammas = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001, 0.000000001, 0.0000000001, 0.0]
    elif hyperparametersRange == "onlyGamma":
        allNus =    [0.0]
        allGammas = [float(2 ** (-i)) for i in range(30)]
        allGammas.append(0)
                
    elif hyperparametersRange == "fine":
        allNus =    [0.0003, 0.0001, 0.00007, 0.00005, 0.0005]
        allGammas = [0.01, 0.007, 0.005, 0.05, 0.0001, 0.00001]
    elif hyperparametersRange == "onlyNu":
        allNus = [float(2 ** (-i)) for i in numpy.arange(0, 30, 0.1)] # setting in journal
        allNus.append(0)
        allGammas = [0.0]
      
    elif hyperparametersRange == "onlyNuKM":
        assert(False)
        allNus =    [0.0, 0.00005, 0.00001, 0.000001, 0.0000001, 0.00000001]
        allGammas = [0.0]
    else:
        assert(False)
        
    return allNus, allGammas


def projectData(dataFeatures, sortedClusters):

    OLD_DIM = dataFeatures.shape[1]
    NEW_DIM = len(sortedClusters)

    T = numpy.zeros((NEW_DIM, OLD_DIM))
    
    for i in range(NEW_DIM):
        for j in sortedClusters[i]:
            T[i,j] = 1
    
    return numpy.dot(dataFeatures, T.transpose())

        


def getTwoSplit(allCovariatesAsMatrix, allLabels):
    NUMBER_OF_FOLDS = 2
    kfoldSplitter = sklearn.model_selection.StratifiedKFold(n_splits=NUMBER_OF_FOLDS, random_state=432532, shuffle=True)
    for train_index, test_index in kfoldSplitter.split(allCovariatesAsMatrix, allLabels):
        return allCovariatesAsMatrix[train_index], allLabels[train_index], allCovariatesAsMatrix[test_index], allLabels[test_index]
                 
    
def calculateSquaredDistanceMatrix(completeWEMatrix):
    numberOfAllWords = completeWEMatrix.shape[0]
    distanceMatrix = numpy.zeros(shape = (numberOfAllWords, numberOfAllWords), dtype = completeWEMatrix.dtype)
    
    for wordId in range(numberOfAllWords):
        diffToAllWords = completeWEMatrix - completeWEMatrix[wordId,]
        distanceMatrix[wordId, ] = math_helpers.dotProdOfEachRow(diffToAllWords)
     
    return distanceMatrix


def setToZeroOutsideClusters(similarityMatrix, clusteringLabels):
    NR_FEATURES = similarityMatrix.shape[0]
    assert(NR_FEATURES == similarityMatrix.shape[1] and NR_FEATURES == len(clusteringLabels))
    
    for i in range(NR_FEATURES):
        for j in range(i+1,NR_FEATURES,1):
            if clusteringLabels[i] != clusteringLabels[j]:
                similarityMatrix[i,j] = 0.0
                similarityMatrix[j,i] = 0.0

    return


def thresholdSimilarityMatrix(covariateSims, avgNeighbours):
    
    if avgNeighbours == "all":
        return
    
    NUMBER_OF_COVARIATES = covariateSims.shape[0]
    
    uniqueSims = covariateSims[numpy.triu_indices(NUMBER_OF_COVARIATES, k=1)] # take the upper half without diagonal
    
    totalNumberDesiredNeighbours = avgNeighbours * NUMBER_OF_COVARIATES
    threshold = numpy.sort(uniqueSims)[- int(totalNumberDesiredNeighbours * 0.5)] # need to half because we took only one half of the symmetric matrix
    
    covariateSims[covariateSims < threshold] = 0.0
    print("threshold = ", threshold)
    
    totalNumberOfNeighbours = 0
    
    for i in range(NUMBER_OF_COVARIATES):
        adjacentNodes = []
        for j in range(NUMBER_OF_COVARIATES):
            if i != j and covariateSims[i,j] > 0.0:
                adjacentNodes.append(j)
       
        totalNumberOfNeighbours += len(adjacentNodes)
    
    print("average number of neighbours = ", (float(totalNumberOfNeighbours) / float(NUMBER_OF_COVARIATES)))
    return
     


def getBaseFilename(CORPUS_NAME, DATA_SPECIFIER_STRING):
    return PATH_TO_SOURCE_FOLDER + "trainingResults/" + CORPUS_NAME + "_" + DATA_SPECIFIER_STRING + "_" 

def getResultStatisticsFilename(DATA_NAME, hyperparametersRange, EVAL_CRITERIA, WITH_RETRAINING, additionalInfo):
    FOLDER_NAME = PATH_TO_SOURCE_FOLDER + "evalResults/" 
    filename = FOLDER_NAME + DATA_NAME + "_" + hyperparametersRange + "_" + EVAL_CRITERIA + "_" + str(WITH_RETRAINING) 
    if len(additionalInfo) > 0:
        return filename + "_" + additionalInfo
    else:
        return filename

def getResultStatisticsKMeansFilename(DATA_NAME, EVAL_CRITERIA, additionalInfo):
    FOLDER_NAME = PATH_TO_SOURCE_FOLDER + "evalResults/" 
    filename = FOLDER_NAME + DATA_NAME + "_" + EVAL_CRITERIA 
    if len(additionalInfo) > 0:
        return filename + "_" + additionalInfo
    else:
        return filename


# allUniqueClusterings and allUniqueRelevances are lists
def getMatchingId(allUniqueClusterings, allUniqueRelevances, clusteringIds, relevanceIds):
    assert(len(allUniqueClusterings) == len(allUniqueRelevances))
    
    for registeredId in range(len(allUniqueClusterings)):
        registeredClustering = allUniqueClusterings[registeredId] 
        registeredRelevance = allUniqueRelevances[registeredId]
        matchingScore = sklearn.metrics.adjusted_mutual_info_score(registeredClustering, clusteringIds)
        print("matchingScore = ", matchingScore)
        if (matchingScore == 1.0) and numpy.all(registeredRelevance == relevanceIds):
            return registeredId
    
    assert(False)
    return None


# e.g.  clusteringSetRep = [{0,2}, {3,1,4}] becomes [0 1 0 1 1]
def getListRep(clusteringSetRep):
    if len(clusteringSetRep) == 0:
        return numpy.zeros(0, dtype = numpy.int)
    
    largestElem = 0
    nrElems = 0 
    for cluster in clusteringSetRep:
        largestElem = max([largestElem, max(cluster)])
        nrElems += len(cluster)
    
#     if (largestElem != nrElems - 1):
#         print(clusteringSetRep)
#         print("len(clusteringSetRep) = ", len(clusteringSetRep))
#         print("largestElem = ", largestElem)
    assert(largestElem == nrElems - 1)
    
    clusteringListRep = numpy.ones(nrElems, dtype = numpy.int) * -1
    
    for clusterId, cluster in enumerate(clusteringSetRep):
        clusteringListRep[list(cluster)] = clusterId
        
    assert(numpy.all(clusteringListRep >= 0))
    return clusteringListRep


# e.g.  clusteringSetRep = [{0,2}, {3,1,4}] becomes {0,2,3,1,4}
def getOneSetRep(clusteringSetRep):
    
    oneSet = set()
    
    for cluster in clusteringSetRep:
        assert(len(cluster) == 1)
        assert(len(cluster & oneSet) == 0)
        oneSet = oneSet | cluster

    return oneSet


def existsSelectedFeatures(allClusterings, newClustering):
    for clustering in allClusterings:
        if (len(clustering) == len(newClustering)) and getOneSetRep(clustering).issubset(getOneSetRep(newClustering)):
            return True
    
    return False

def existsClusteringInList(allClusterings, newClustering):
    
    for clustering in allClusterings:
        if (len(clustering) == len(newClustering)) and (sklearn.metrics.adjusted_mutual_info_score(getListRep(clustering), getListRep(newClustering)) == 1.0):
            return True
    
    return False

    
def getUniqueResults(allClusteringsFound, allRelevancesFound, allHyperparametersInOrder):
    
    allUniqueClusterings = []
    allUniqueRelevances = []
    correspondingHyperparameters = []
    
    for i in range(allClusteringsFound.shape[0]):
        clusteringIds = allClusteringsFound[i]
        relevanceIds = allRelevancesFound[i]
        hyperparams = allHyperparametersInOrder[i]
        
        alreadyExists = False
        for registeredId in range(len(allUniqueClusterings)):
            registeredClustering = allUniqueClusterings[registeredId] 
            registeredRelevance = allUniqueRelevances[registeredId]
            if (sklearn.metrics.adjusted_mutual_info_score(registeredClustering, clusteringIds) == 1.0) and numpy.all(registeredRelevance == relevanceIds):
                alreadyExists = True
                correspondingHyperparameters[registeredId].add(hyperparams)
                break
        
        if not alreadyExists:
            allUniqueClusterings.append(clusteringIds)
            allUniqueRelevances.append(relevanceIds)
            newHyperparamsSet = set()
            newHyperparamsSet.add(hyperparams)
            correspondingHyperparameters.append(newHyperparamsSet)
                
    return allUniqueClusterings, allUniqueRelevances, correspondingHyperparameters

def getDataSpecifierForText(NR_FEATURES, TRAINING_DATA_SIZE, sampleId, WORD_EMBEDDING_TYPE):
    return str(TRAINING_DATA_SIZE) + "size_s" + str(sampleId) + "_" + str(NR_FEATURES) + "features" + "_" + WORD_EMBEDDING_TYPE
    



def saveFilteredDims(DATA_NAME, allDimsSet):
     
    filename = FILTERED_DIMS_FILENAME_STEM + "_" + DATA_NAME + ".txt"
    
       
    allSelectedDimsAsList = []
    
    for dimSet in allDimsSet:
        assert(len(dimSet) == 1)
        elem = list(dimSet)[0]
        allSelectedDimsAsList.append(elem)
    
    allSelectedDimsAsList = numpy.sort(allSelectedDimsAsList)
    
    assert(len(allSelectedDimsAsList) == len(allDimsSet))
    # assert(not pathlib.Path(filename).is_file()) # ensure that file does not already exist 
     
    with open(filename, 'w') as f:
        for elem in allSelectedDimsAsList:
            f.write(str(elem) + "\n")
    
    print("sorted filtered dims = ", allSelectedDimsAsList)
    print("SAVED FILTERED DIMS TO " + FILTERED_DIMS_FILENAME_STEM + "_" + DATA_NAME + ".txt")
    return

def loadFilteredDims(DATA_NAME):
 
    allDimIds = []
     
    with open(FILTERED_DIMS_FILENAME_STEM + "_" + DATA_NAME + ".txt", 'r') as allDimsFile:
        for line in allDimsFile:
            dim = line.strip()
            allDimIds.append(int(dim))
     
    return numpy.asarray(allDimIds, dtype = numpy.int)


def loadAllData(CORPUS_NAME, NR_FEATURES, TRAINING_DATA_SIZE, sampleId, WORD_EMBEDDING_TYPE, standardizeCovariates, filterOutWords):
    assert(filterOutWords == True or filterOutWords == False)
    
    FLOAT_TYPE = float

    if CORPUS_NAME == "aptemod":
        allTargetLabels = "earn,acq,money-fx,grain,crude,trade,interest,wheat,ship,corn"
    elif CORPUS_NAME == "newsgroup20all":
        allTargetLabels = "rec.sport.hockey,soc.religion.christian,rec.motorcycles,rec.sport.baseball,sci.crypt,rec.autos,sci.med,comp.windows.x,sci.space,sci.electronics,comp.os.ms-windows.misc,comp.sys.ibm.pc.hardware,misc.forsale,comp.graphics,comp.sys.mac.hardware,talk.politics.mideast,talk.politics.guns,alt.atheism,talk.politics.misc,talk.religion.misc"
    elif CORPUS_NAME == "IMDB":
        allTargetLabels = "goodMovie"
    else:
        assert(False)
            
    allTargetLabels = allTargetLabels.split(",")
    
    if WORD_EMBEDDING_TYPE == "inCorpus":
        EMBEDDING_DIM = 50 # word2vec embedding dim
        ORIGINAL_WES_FILENAME = CORPUS_NAME + "_all_word2vec_" + str(EMBEDDING_DIM) + "_we_plain"
        print("USE ORIGINAL_WES_FILENAME = ", ORIGINAL_WES_FILENAME)
        existingWEs = prepare.loadOriginalWE(prepare.getNormalWEDirectory(CORPUS_NAME) + ORIGINAL_WES_FILENAME)
    else:
        EMBEDDING_DIM = 300 
        _, existingWEs = prepare.loadWordEmbeddingsSimple("glove.6B.300d.txt", usedWords = None, has_LOW_FREQ_WORD_SYMBOL = False)
    
    print("number of words with embeddings = ", len(existingWEs))
    
    DATA_SPECIFIER_STRING = getDataSpecifierForText(NR_FEATURES, TRAINING_DATA_SIZE, sampleId, WORD_EMBEDDING_TYPE)
    
    wordToDimFilename = prepare.getDocRepDirectoryForSparseModeling(CORPUS_NAME) + CORPUS_NAME  + "_" + "bow_wordTodim_" + DATA_SPECIFIER_STRING
    print("wordToDimFilename = ", wordToDimFilename)
    
    wordToDim = io_helper.loadWordToDim(wordToDimFilename)
    print("allWords = ", len(wordToDim))
    
    existingWEs = load_documents.filterAndNormalizeWE(existingWEs, wordToDim.keys())
    dimToWord = {v: k for k, v in wordToDim.items()}
    
    TRAINING_DATA_DOC_REP = prepare.getDocRepDirectoryForSparseModeling(CORPUS_NAME) + CORPUS_NAME  + "_training_data_doc_rep_BOW_" + DATA_SPECIFIER_STRING
    TEST_DATA_DOC_REP = prepare.getDocRepDirectoryForSparseModeling(CORPUS_NAME) + CORPUS_NAME  + "_test_data_doc_rep_BOW_" + DATA_SPECIFIER_STRING
    
    allTrainingTextsAsMatrix, allTrainingLabels = load_documents.loadMultinomialClassLabelTextsAsMatrix(TRAINING_DATA_DOC_REP, len(wordToDim), FLOAT_TYPE)
    allTestTextsAsMatrix, allTestLabels = load_documents.loadMultinomialClassLabelTextsAsMatrix(TEST_DATA_DOC_REP, len(wordToDim), FLOAT_TYPE)
    
    print("total number of training texts = ", allTrainingTextsAsMatrix.shape[0])
    print("total number of test texts = ", allTestTextsAsMatrix.shape[0])
    
    assert(allTrainingTextsAsMatrix.shape[1] == len(wordToDim))
    assert(allTestTextsAsMatrix.shape[1] == len(wordToDim))

    assert(standardizeCovariates)

    if filterOutWords:
        
        allUsedWordIds = loadFilteredDims(CORPUS_NAME)
        allUsedWords = [dimToWord[wordId] for wordId in allUsedWordIds] 
        
        existingWEsNew = {word : existingWEs[word] for word in allUsedWords}
        
        wordToDimNew = {}
        newId = 0
        for word in allUsedWords:
            wordToDimNew[word] = newId
            newId += 1
        
        dimToWordNew = {v: k for k, v in wordToDimNew.items()}
        
        completeWEMatrixNew = load_documents.convertToNumpyMatrix(existingWEsNew, EMBEDDING_DIM, wordToDimNew, FLOAT_TYPE)
        
        allTrainingTextsAsMatrixNew = allTrainingTextsAsMatrix[:, allUsedWordIds]
        allTestTextsAsMatrixNew = allTestTextsAsMatrix[:, allUsedWordIds]
        
        allTrainingTextsAsMatrixNew, allTestTextsAsMatrixNew = standardizeData(allTrainingTextsAsMatrixNew, allTestTextsAsMatrixNew)
        
        assert(allTestTextsAsMatrixNew.shape[1] == len(allUsedWords))
        return allTrainingTextsAsMatrixNew, allTrainingLabels, allTestTextsAsMatrixNew, allTestLabels, completeWEMatrixNew, wordToDimNew, dimToWordNew, DATA_SPECIFIER_STRING
    else:
        
        completeWEMatrix = load_documents.convertToNumpyMatrix(existingWEs, EMBEDDING_DIM, wordToDim, FLOAT_TYPE)
        
        allTrainingTextsAsMatrix, allTestTextsAsMatrix = standardizeData(allTrainingTextsAsMatrix, allTestTextsAsMatrix)

        return allTrainingTextsAsMatrix, allTrainingLabels, allTestTextsAsMatrix, allTestLabels, completeWEMatrix, wordToDim, dimToWord, DATA_SPECIFIER_STRING


    
    

def showAvgAndStd(resultArray, digits = 2):
    return str(round(numpy.average(resultArray),digits)) + " (" + str(round(numpy.std(resultArray),digits)) + ") " 

# checked
def blockThreshold(a, v):
    zeroVec = numpy.zeros_like(v)
    mag = l2Norm(v)
    if mag <= 0.0:
        return zeroVec
    else:
        t = 1.0 - (a / mag)
        if t <= 0.0:
            return zeroVec
        else:
            return t * v 

# return True iff "blockThreshold" returns 0 vector
def isBlockThresholdedToZero(a, v):
    mag = l2Norm(v)
    if mag <= 0.0:
        return True
    else:
        t = 1.0 - (a / mag)
        return(t <= 0.0)


def l2Norm(v):
    return numpy.sqrt(v.dot(v))

def square(v):
    return v.dot(v)

def getVecStr(vec):
    s = "["
    for i in range(vec.shape[0]):
        s += " " + str(round(vec[i],3)) 
    s += " ]"
    return s

def getStatsStr(vec):
    return str(round(numpy.mean(vec),2)) + " (" + str(round(numpy.std(vec),2)) + ") " 
    
def showVec(vec):
    assert(len(vec.shape) == 1)
    print(getVecStr(vec))
    return

def showVecInt(vec):
    s = "["
    for i in range(vec.shape[0]):
        s += " " + str(vec[i]) 
    s += " ]"
    print(s)
    return

def showMatrix(m):
    mArray = numpy.asarray(m)
    assert(len(mArray.shape) == 2)
    s = "["
    for i in range(mArray.shape[0]):
        s += getVecStr(mArray[i]) + "\n"
    s = s.strip()
    s += "]"
    print(s)
    return 

def show(a):
    assert(len(a.shape) == 1 or len(a.shape) == 2)
    if len(a.shape) == 1:
        showVec(a)
    else:
        showMatrix(a)
        
    return



def getPredictions(B, b0, dataFeatures, dataLabels):
    assert(type(B) == numpy.matrixlib.defmatrix.matrix)
    assert(type(b0) == numpy.matrixlib.defmatrix.matrix)
    assert(type(dataFeatures) == numpy.matrixlib.defmatrix.matrix)
     
    n = dataFeatures.shape[0]
    predLogLikelihoods = numpy.zeros(n)
    predLabels = numpy.zeros(n)
    for s in range(n):
        predLogLikelihoods[s] = (B[dataLabels[s],:] * dataFeatures[s,:].T + b0[dataLabels[s]]) - scipy.misc.logsumexp(B * dataFeatures[s,:].T + b0)
        predLabels[s] = numpy.argmax(B * dataFeatures[s,:].T + b0)    
    
    return predLabels, predLogLikelihoods



# checked
def getAccuracy(predLabels, dataLabels):
    
    n = predLabels.shape[0]
     
    totalCorrect = 0
   
    for s in range(n):
        predictedClass = predLabels[s]
        assert(predictedClass >= 0.0 and predictedClass <= 100)
        if predictedClass == dataLabels[s]:
            totalCorrect += 1.0
    
    accuracy = float(totalCorrect) / float(n)
    return accuracy

# checked
def evalData(predLabels, predLogLikelihoods, dataLabels):
    
    n = predLabels.shape[0]
    k = dataLabels.max() + 1
     
    logLikelihood = 0
    totalCorrect = 0
    predClassCounts = numpy.zeros(k)    
    
    for s in range(n):
        logLikelihood += predLogLikelihoods[s]
        predictedClass = predLabels[s]
        assert(predictedClass >= 0.0 and predictedClass <= 100)
        predClassCounts[int(predictedClass)] += 1.0
        # print "predictedClass = ", predictedClass
        if predictedClass == dataLabels[s]:
            totalCorrect += 1.0
    
    accuracy = float(totalCorrect) / float(n)
    predClassDist = predClassCounts / float(n)
    
    # print "logLikelihood = ", logLikelihood
    # print "accuracy = ", accuracy
    # print "predicated class distribution = ", predClassDist
    return accuracy, logLikelihood



# checked
def countRelevantFeatures(clusterIds, relevance):
    assert(len(clusterIds) == len(relevance))
    assert(numpy.min(clusterIds) == 1)
    
    relevantClusterIds = set()
    
    for i in range(0, len(clusterIds), 1):
        if relevance[i] == 1:
            relevantClusterIds.add(clusterIds[i])

    return len(relevantClusterIds)

    
def getClusterIds(clusterSets):
    
    allCovariates = set()
    for s in clusterSets:
        allCovariates = allCovariates | s
    
    clusterIdsVec = numpy.zeros(len(allCovariates), dtype = numpy.int)
    for clusterId, cluster in enumerate(clusterSets):
        for covariateId in cluster:
            clusterIdsVec[covariateId] = clusterId
    
    return clusterIdsVec

def evaluateAll(trueClusterIds, trueRelevance, clusteringInfo):
    
    outputClusterIds, outputRelevance = clusteringInfo
    
    adjustedNMI = sklearn.metrics.adjusted_mutual_info_score(trueClusterIds, outputClusterIds)
    featureSelectionF1 = sklearn.metrics.f1_score(trueRelevance, outputRelevance)

#     print "RESULT EVALUATION:"
#     print "true clusterIds = "
#     showVecInt(trueClusterIds)
#     print "prop clusterIds = "
#     showVecInt(outputClusterIds)
#     print "selected features = "
#     showVecInt(outputRelevance)
#     print "adjustedNMI = ", adjustedNMI
#     print "featureSelectionF1 = ", featureSelectionF1
#     print "number of selected covariates = ", numpy.sum(outputRelevance)
    
    return (adjustedNMI, featureSelectionF1)



def getRelevantClusterList(clusteringInfo):
    allClusterIds, allRelevanceIds = clusteringInfo
    
    relevantClusterIds = set(allClusterIds)
    
    origNrIrrelevantFeatures = numpy.count_nonzero(allRelevanceIds == 0)
    if origNrIrrelevantFeatures > 0:
        IRRELEVANT_CLUSTER_ID = numpy.max(allClusterIds)
        relevantClusterIds.remove(IRRELEVANT_CLUSTER_ID)
    
   
    clustersInOrder = []
    for clusterId in relevantClusterIds:
        clustersInOrder.append(set(numpy.where(allClusterIds == clusterId)[0]))
    
    allClusterSizes = numpy.zeros(len(clustersInOrder))
    for clusterId in range(len(clustersInOrder)):
        allClusterSizes[clusterId] = len(clustersInOrder[clusterId])
    
    sortedClusters = []
    for i in numpy.argsort(allClusterSizes):
        sortedClusters.append(clustersInOrder[i])
    
#     print "cluster histogram:"
#     for i, cluster in enumerate(sortedClusters):
#         print "cluster " + str(i) + " has size = " + str(len(cluster))
#         print "feature ids = ", cluster
     
    # sanity check:  
    totalNrSelectedFeatures = 0
    for cluster in sortedClusters:
        clusterSize = len(cluster)
        assert(clusterSize >= 1)
        totalNrSelectedFeatures += clusterSize
    assert(totalNrSelectedFeatures == numpy.sum(allRelevanceIds))
    
    print("number of relevant features = ", numpy.sum(allRelevanceIds))
    print("number of relevant clusters = ", len(sortedClusters))
    return sortedClusters


def showCluster(dimToWord, cluster):
    assert(len(cluster) >= 1)
    
    allWords = ""
    for i in cluster:
        allWords += dimToWord[i] + ", "
    
    print(allWords)

def showAllClustersContent(dimToWord, sortedClusters):
    print("ALL CLUSTERS CONTENT:")
    for clusterId in range(len(sortedClusters)):
        cluster = sortedClusters[clusterId]
        print("Cluster " + str(clusterId) + " (size = " + str(len(cluster)) + " ):") 
        showCluster(dimToWord, cluster)

def showAllClustersContentLatex(dimToWord, sortedClusters):
    
    MAX_CLUSTERS_SHOW = 34
    MAX_WORDS_SHOW = 15
    
    print("ALL CLUSTERS CONTENT:")
    for clusterId in range(len(sortedClusters)-1, len(sortedClusters)-MAX_CLUSTERS_SHOW-1, -1):
        cluster = sortedClusters[clusterId]
        
        MAX_WORDS_SHOW = 15
        allWordsAsList = [dimToWord[i] for i in cluster]
        if len(cluster) > MAX_WORDS_SHOW:
            allWordsAsList = allWordsAsList[0:MAX_WORDS_SHOW]
            allWordsAsList.append("...(and more)...")
          
        print("{ \\bf Cluster " + str(clusterId) + " (size = " + str(len(cluster)) + " ): }")
        print(",".join(allWordsAsList) + " \\\\" )
        print("\\midrule")

def standardizeDataWithOneDesignMatrix(designMatrix):
    allMeans = numpy.mean(designMatrix, axis = 0)
    allStds = numpy.std(designMatrix, axis = 0)
     
    zeroStdsCovariates = numpy.where(allStds == 0.0)[0]
    if len(zeroStdsCovariates) > 0:
        print("zeroStdsCovariates = ", zeroStdsCovariates)
        print("nr = ", len(zeroStdsCovariates))
     
    assert(numpy.all(allStds != 0.0))
    return standardizeDataWithMeanStd(allMeans, allStds, designMatrix)


def standardizeDataWithMeanStd(allMeans, allStds, designMatrix):
    standardizedDesignMatrix = designMatrix - allMeans
    standardizedDesignMatrix = standardizedDesignMatrix / allStds
    return standardizedDesignMatrix

def standardizeData(designMatrixTrain, designMatrixTest):
    designMatrixFull = numpy.vstack([designMatrixTrain, designMatrixTest])
    print("designMatrixFull.shape = ", designMatrixFull.shape)
    allMeans = numpy.mean(designMatrixFull, axis = 0)
    allStds = numpy.std(designMatrixFull, axis = 0)
    
    zeroStdsCovariates = numpy.where(allStds == 0.0)[0]
    if len(zeroStdsCovariates) > 0:
        print("WARNING ZERO STANDARD DEVIATIONS")
        print("zeroStdsCovariates = ", zeroStdsCovariates)
        print("nr = ", len(zeroStdsCovariates))
        allStds[zeroStdsCovariates] = numpy.inf
        
    assert(numpy.all(allStds != 0.0))
    return standardizeDataWithMeanStd(allMeans, allStds, designMatrixTrain), standardizeDataWithMeanStd(allMeans, allStds, designMatrixTest)
