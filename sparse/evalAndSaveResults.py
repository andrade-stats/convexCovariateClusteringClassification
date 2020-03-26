
import sys
import numpy
import helper
import bayesianClusterEvaluationLaplacian
import experiments
import baselineMultiLogReg
import pickle
import syntheticDataGeneration
import sklearn.metrics
import showResultsSyntheticData
import constants


# example:
# /opt/intel/intelpython3/bin/python evalAndSaveResults.py SYNTHETIC_DATA onlyNu 10 all smallFullContra 10


numpy.random.seed(3523421)

DATA_NAME = sys.argv[1]
hyperparametersRange = sys.argv[2]
TOTAL_NUMBER_OF_FOLDS = int(sys.argv[3])

if DATA_NAME != "SYNTHETIC_DATA" and len(sys.argv) >= 6:
    
    # ANALYZE TEXT DATA

    assert(TOTAL_NUMBER_OF_FOLDS == 1)
    
    TRAINING_DATA_SIZE = int(sys.argv[4])
    assert(TRAINING_DATA_SIZE >= 100)
    
    if sys.argv[5] == "all":
        avgNeighbours = "all"
    else:
        avgNeighbours = float(sys.argv[5])
        assert(avgNeighbours > 0.0 and avgNeighbours <= 2.0)
    
    if DATA_NAME == "IMDB":
        WORD_EMBEDDING_TYPE = "inCorpus"
    elif DATA_NAME == "newsgroup20all":    
        WORD_EMBEDDING_TYPE = "glove.6B.300d"
    else:
        assert(False)
    
    assert(WORD_EMBEDDING_TYPE == "inCorpus" or WORD_EMBEDDING_TYPE == "glove.6B.300d")
    
    NR_FEATURES = 10000
    standardizeCovariates = True
    TEXT_DATA = True
    
    
    TRAIN_DATA_SEPCIFIER = "_" + str(TRAINING_DATA_SIZE) + "train"
        
else:
    
    # ANALYZE NON-TEXT DATA REAL OR SYNTHETIC
    
    assert(DATA_NAME != "newsgroup20all" and DATA_NAME != "IMDB")
    
    if sys.argv[4] == "all":
        avgNeighbours = "all"
    else:
        avgNeighbours = float(sys.argv[4])
        assert(avgNeighbours > 0.0 and avgNeighbours <= 2.0)
    
    TEXT_DATA = False   
    TRAIN_DATA_SEPCIFIER = ""




if DATA_NAME == "SYNTHETIC_DATA":
    assert(TOTAL_NUMBER_OF_FOLDS == 10)
    testSetting = sys.argv[5]
    NUMBER_OF_SAMPLES_PER_CLASS = int(sys.argv[6])
    assert(NUMBER_OF_SAMPLES_PER_CLASS >= 10)
    covariateSims, dataFeature_allFolds, dataLabels_allFolds, trueClusterIds, trueRelevantCovariates, NUMBER_OF_CLASSES, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS = syntheticDataGeneration.generateData(testSetting, NUMBER_OF_SAMPLES_PER_CLASS, TOTAL_NUMBER_OF_FOLDS)

   
if avgNeighbours == "all":
    filterOutDims = True
else:
    filterOutDims = False

    
if hyperparametersRange == "onlyGamma":
    assert(avgNeighbours == 0.5)
elif hyperparametersRange == "onlyNu":
    assert(avgNeighbours == "all")
else:
    assert(False)


assert(hyperparametersRange == "onlyGamma" or hyperparametersRange == "onlyNu")
if hyperparametersRange == "onlyGamma":
    assert(not filterOutDims)
else:
    assert(filterOutDims)
    

optParams = {}
optParams["AVG_NEIGHBOURS"] = avgNeighbours
optParams["INITIAL_RHO"] = 1.0
optParams["MAX_LBFGS_ITERATIONS"] = 15000 # default value
optParams["ADMM_MAX_ITERATIONS"] = 1000 # default value
optParams["RHO_MULTIPLIER"] = 1.0 # default value
optParams["EPSILON"] = 0.00001 # default value

# optParams["CVXPY_USED"] = "_trainedWithCVXPY"
optParams["CVXPY_USED"] = ""
optParams["SOLVER"] = ""

allNus, allGammas = helper.getHyperparameters(hyperparametersRange)
TOTAL_NUMBER_OF_PARAMETERS = len(allNus) * len(allGammas)

for foldId in range(TOTAL_NUMBER_OF_FOLDS):

    if TEXT_DATA:
        allTrainingDataAsMatrix, allTrainingLabels, allTestDataAsMatrix, allTestLabels, completeWEMatrix, wordToDim, dimToWord, DATA_SPECIFIER_STRING = helper.loadAllData(DATA_NAME, NR_FEATURES, TRAINING_DATA_SIZE, foldId, WORD_EMBEDDING_TYPE, standardizeCovariates, filterOutDims)
    elif DATA_NAME == "SYNTHETIC_DATA":
       
        DATA_SPECIFIER_STRING = testSetting + "_" + str(NUMBER_OF_SAMPLES_PER_CLASS) + "sc"  + "_" + str(foldId)
        TRAIN_DATA_SEPCIFIER = testSetting + "_" + str(NUMBER_OF_SAMPLES_PER_CLASS) + "sc"
        
        allTrainingDataAsMatrix = dataFeature_allFolds[foldId]
        allTrainingLabels = dataLabels_allFolds[foldId]
        allTestDataAsMatrix = None
        allTestLabels = None
        
    else:
        allTrainingDataAsMatrix, allTrainingLabels, allTestDataAsMatrix, allTestLabels, numberOfClasses, similarityMatrix = realDataHelper.loadRealData(DATA_NAME, foldId, filterOutDims)
        DATA_SPECIFIER_STRING = str(foldId)

    BASE_FILENAME = helper.getBaseFilename(DATA_NAME, DATA_SPECIFIER_STRING)
    
    allResultsId = 0
    allResultsWithEmptyRows = numpy.zeros(shape = (TOTAL_NUMBER_OF_PARAMETERS, 9))
    
    print("TOTAL_NUMBER_OF_PARAMETERS = ", TOTAL_NUMBER_OF_PARAMETERS)
    
    print("DATA SATISTICS:")
    print("n = ", allTrainingDataAsMatrix.shape[0])
    print("p = ", allTrainingDataAsMatrix.shape[1])
    
    bestLambdaParamFullModel, _ = baselineMultiLogReg.runLogisticRegressionCVnew(allTrainingDataAsMatrix, allTrainingLabels, "accuracy")
    print("best sigma = ", numpy.sqrt(1.0 / (2.0 * bestLambdaParamFullModel)))
    # assert(False)
    
    assert(numpy.max(allTrainingLabels) >= 1 and numpy.min(allTrainingLabels) == 0)
    
    allClusterings = []
    allClustersClassWeights = []
    
    for nuId, nu in enumerate(allNus):
        for gammaId, gamma in enumerate(allGammas):
                        
            partialConnectedInfo, _, _ = experiments.loadStatistics(BASE_FILENAME, optParams, nu, gamma)
            
            sortedClusters = helper.getRelevantClusterList(partialConnectedInfo)
            effectiveNumberOfFeatures = len(sortedClusters)
            
            if hyperparametersRange == "onlyGamma":
                # we actually do only feature selection
                if helper.existsSelectedFeatures(allClusterings, sortedClusters):
                    continue
            else:
                # we actually do clustering
                if helper.existsClusteringInList(allClusterings, sortedClusters):
                    continue
            
            
            logMargLaplacianDiag, B_MAP, sigma, trainCVlogProb, trainCVacc = bayesianClusterEvaluationLaplacian.evalClusteringNew(allTrainingDataAsMatrix, allTrainingLabels, sortedClusters, False, bestLambdaParamFullModel)
            
            if allTestDataAsMatrix is not None:
                trainAcc, testAcc = baselineMultiLogReg.evalModelOnTrainAndTestDataNew(allTrainingDataAsMatrix, allTrainingLabels, sortedClusters, allTestDataAsMatrix, allTestLabels, sigma)
                ANMI = -1
            else:
                # dataFeatures = helper.projectData(allTrainingDataAsMatrix, sortedClusters)
                # _, trainCVAcc = baselineMultiLogReg.runLogisticRegressionCVnew(dataFeatures, allTrainingLabels, "accuracy")
                trainAcc = -1
                testAcc = -1
                ANMI, _ = helper.evaluateAll(trueClusterIds, trueRelevantCovariates, partialConnectedInfo)
                
                # reevalANMI = sklearn.metrics.adjusted_mutual_info_score(trueClusterIds, helper.getClusterIds(sortedClusters))
                # print(sortedClusters)
                # print(helper.getClusterIds(sortedClusters))
                # print(trueClusterIds)
                # print("ANMI = ", ANMI)
                # print("reevalANMI = ", reevalANMI)
                # assert(numpy.abs(reevalANMI - ANMI) < 0.0001)
                # assert(False)
                
                
            assert(B_MAP.shape[1] == len(sortedClusters))
            allClusterings.append(sortedClusters)
            allClustersClassWeights.append(B_MAP)
            
            allResultsWithEmptyRows[allResultsId, constants.NU_ID] = nu
            allResultsWithEmptyRows[allResultsId, constants.GAMMA_ID] = gamma
            allResultsWithEmptyRows[allResultsId, constants.EFFECTIVE_NUMBER_OF_COVARIATES_ID] = effectiveNumberOfFeatures
            allResultsWithEmptyRows[allResultsId, constants.LOG_MARGINAL_LAPLACE_DIAG_VALIDATION_CRITERIA_ID] = logMargLaplacianDiag
            allResultsWithEmptyRows[allResultsId, constants.TRAIN_ACCURACY_ID] = trainAcc
            allResultsWithEmptyRows[allResultsId, constants.TEST_ACCURACY_ID] = testAcc
            allResultsWithEmptyRows[allResultsId, constants.TRAIN_CV_LOG_PROB_ID] = trainCVlogProb
            allResultsWithEmptyRows[allResultsId, constants.TRAIN_CV_ACC_ID] = trainCVacc
            allResultsWithEmptyRows[allResultsId, constants.ANMI_ID] = ANMI
            allResultsId += 1
     
     
    print("allResultsId = ", allResultsId)
    assert(len(allClusterings) == allResultsId)
    allResults = allResultsWithEmptyRows[0:allResultsId]
    
    filename = helper.EVALUATION_RESULTS_FOLDER + DATA_NAME + TRAIN_DATA_SEPCIFIER + "_" + hyperparametersRange + "_" + str(avgNeighbours) + "Neighbours_" + str(foldId) + "fold"
    numpy.save(filename, allResults)
    with open(filename + "_clusterings.pkl", "wb") as f:
        pickle.dump(allClusterings, f)
    with open(filename + "_clusteringsClassWeights.pkl", "wb") as f:
        pickle.dump(allClustersClassWeights, f)
    
    print("SAVED RESULTS STATISTICS TO FILE: " + filename)

 


print("FINISHED SAVING ALL RESULTS")

if DATA_NAME == "SYNTHETIC_DATA":
    print("EVALUATION RESULTS SUMMARY:")
    showResultsSyntheticData.showResults(testSetting, NUMBER_OF_SAMPLES_PER_CLASS, hyperparametersRange)
