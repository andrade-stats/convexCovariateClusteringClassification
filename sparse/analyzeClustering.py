import numpy
import pickle
import helper
import visualizer
import constants


DATA_NAME = "SYNTHETIC_DATA"
testSetting = "smallContra"
NUMBER_OF_SAMPLES_PER_CLASS = 100


hyperparametersRange = "onlyNu"
avgNeighbours = "all"

foldId = 0


TRAIN_DATA_SEPCIFIER = testSetting + "_" + str(NUMBER_OF_SAMPLES_PER_CLASS) + "sc"



criteriaId = constants.LOG_MARGINAL_LAPLACE_DIAG_VALIDATION_CRITERIA_ID


FOLDER_NAME = "../cleanResults/" 
filename = FOLDER_NAME + DATA_NAME + TRAIN_DATA_SEPCIFIER + "_" + hyperparametersRange + "_" + str(avgNeighbours) + "Neighbours_" + str(foldId) + "fold"
       
allResults = numpy.load(filename + ".npy")

if hyperparametersRange == "kMeansClustering":
    
    # need to load with 'latin1' due to Python 2/3 incompatibality
    with open(filename + "_clusterings.pkl", "rb") as f:
        allClusteringsOriginal = pickle.load(f, encoding='latin1')
    
    with open(filename + "_clusteringsClassWeights.pkl", "rb") as f:
        allClustersClassWeightsOriginal = pickle.load(f, encoding='latin1')

else:
    with open(filename + "_clusterings.pkl", "rb") as f:
        allClusteringsOriginal = pickle.load(f)
    
    with open(filename + "_clusteringsClassWeights.pkl", "rb") as f:
        allClustersClassWeightsOriginal = pickle.load(f)



assert(allResults.shape[0] == len(allClusteringsOriginal))
assert(len(allClusteringsOriginal) == len(allClustersClassWeightsOriginal))


if DATA_NAME == "IMDB":
    # set to improve readability
    VISUALIZATION_ODDS_RATIO_THRESHOLD = 1.1
    
    # set to match the number of clusters of the best clustering found by proposed method
    DESIRED_NUMBER_OF_CLUSTERS_KMEANS = 140
    DESIRED_NUMBER_OF_CLUSTERS_CVXCLUSTERING = 85
    allClassLabels = realDataHelper.getClassLabels(DATA_NAME)
elif DATA_NAME == "newsgroup20all":
    # set to improve readability (1.0 means no thresholding)
    VISUALIZATION_ODDS_RATIO_THRESHOLD = 1.0
    
    # set to match the number of clusters of the best clustering found by proposed method
    DESIRED_NUMBER_OF_CLUSTERS_KMEANS = 1174
    DESIRED_NUMBER_OF_CLUSTERS_CVXCLUSTERING = 1219
    allClassLabels = realDataHelper.getClassLabels(DATA_NAME)
elif DATA_NAME == "SYNTHETIC_DATA":
    allClassLabels = ["CLASS A", "CLASS B", "CLASS C", "CLASS D"] 
    VISUALIZATION_ODDS_RATIO_THRESHOLD = 1.0
    dimToWord = {}
    for i in range(1000):
        dimToWord[i] = "covariate" + str(i) 
else:
    assert(False)

# USE THIS TO REDUCE THE OUTPUT GRAPH (for very large graphs which Graphiz cannot handle)
allClusterings, allOddsRatios, allBestClassLabels = visualizer.filterOutCovariates(allClusteringsOriginal, allClustersClassWeightsOriginal, allClassLabels, VISUALIZATION_ODDS_RATIO_THRESHOLD)



if hyperparametersRange == "kMeansClustering":
    METHOD_NAME = "kMeansClustering"
    
    selectedClusteringId = numpy.where(allResults[:, constants.EFFECTIVE_NUMBER_OF_COVARIATES_ID] == DESIRED_NUMBER_OF_CLUSTERS_KMEANS)[0][0]
    allRootNodes = visualizer.getFlatStructure(selectedClusteringId, allClusterings, allOddsRatios, allBestClassLabels)
    
elif hyperparametersRange == "convexClustering":
    METHOD_NAME = "convexClustering"
    
    # go till the number of clusters is DESIRED_NUMBER_OF_CLUSTERS
    lastId = numpy.where(allResults[:, constants.EFFECTIVE_NUMBER_OF_COVARIATES_ID] == DESIRED_NUMBER_OF_CLUSTERS_CVXCLUSTERING)[0][0]
    allResults = allResults[0:(lastId+1),:]
    allClusterings = allClusterings[0:(lastId+1)]
    allOddsRatios = allOddsRatios[0:(lastId+1)]
    allBestClassLabels = allBestClassLabels[0:(lastId+1)]
    
    assert(allResults.shape[0] == len(allClusterings))
    assert(len(allClusterings) == len(allOddsRatios))
    assert(len(allClusterings) == len(allBestClassLabels))
        
    idsForSorting = numpy.arange(lastId, -1, -1)
    allRootNodes = visualizer.getHierarchicalStructure(idsForSorting, allClusterings, allOddsRatios, allBestClassLabels)
    
else:
    assert(hyperparametersRange == "onlyNu")
    METHOD_NAME = "proposed"
    
    idsForSorting = numpy.argsort(- allResults[:, criteriaId])
    
    print("number of roots proposed method (before filtering) = ", len(allClusteringsOriginal[idsForSorting[0]]))
    # assert(len(allClusteringsOriginal[idsForSorting[0]]) == DESIRED_NUMBER_OF_CLUSTERS_KMEANS)
    
    # selectedClusteringId = numpy.where(allResults[:, constants.EFFECTIVE_NUMBER_OF_COVARIATES_ID] == DESIRED_NUMBER_OF_CLUSTERS_PROPOSED)[0][0]
    # assert(len(allClusteringsOriginal[selectedClusteringId]) == DESIRED_NUMBER_OF_CLUSTERS_PROPOSED)
    # place "selectedClusteringId" at front in idsForSorting
    # positionInIdsForSorting = numpy.where(idsForSorting == selectedClusteringId)[0][0]
    # idsForSorting = numpy.delete(idsForSorting, positionInIdsForSorting)
    # idsForSorting = numpy.hstack(([selectedClusteringId], idsForSorting))
    
    allRootNodes = visualizer.getHierarchicalStructure(idsForSorting, allClusterings, allOddsRatios, allBestClassLabels)


for showClassId in range(len(allClassLabels)):
    outputFilename = DATA_NAME + "_" + METHOD_NAME + "_"  + visualizer.convertToWindowsFriendlyName(allClassLabels[showClassId])
    visualizer.showTreeAsGraph(allRootNodes, dimToWord, allClassLabels, showClassId, outputFilename)


visualizer.drawColorLegend(DATA_NAME, allClassLabels)

print("total number of root clusters = ", len(allRootNodes))

print("FINISHED VISUALIZATION")

