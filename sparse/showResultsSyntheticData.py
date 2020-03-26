import sys
import numpy
import helper
import bayesianClusterEvaluationLaplacian
import experiments
import baselineMultiLogReg
import pickle
import syntheticDataGeneration
import sklearn.metrics
import constants


def getANMI(allResults, criteriaID):
    bestId = numpy.argmax(allResults[:, criteriaID])
    return allResults[bestId, constants.ANMI_ID]


def showDetails(trueClusterIds, allClusterings, allResults, criteriaID):
    idsForSorting = numpy.argsort(- allResults[:, criteriaID])
    outputForLatex = ""
    for nr, i in enumerate(idsForSorting):
        
        if nr == 0:
            helper.showVecInt(numpy.asarray(allClusterings[i]))
                
        if nr < 5:
            outputForLatex += " & " + str(round(allResults[i, constants.ANMI_ID],1)) + " & " +  str(int(allResults[i, constants.EFFECTIVE_NUMBER_OF_COVARIATES_ID])) + " & " + str(allResults[i, criteriaID]) + " \\\\ "  + "\n"
        else:
            print("outputForLatex = ")
            print(outputForLatex)
            break
     
    bestId = numpy.argmax(allResults[:, constants.ANMI_ID])
    # print("RESULT WITH BEST ANMI = " + str(round(allResults[bestId, ANMI_ID],1)) + " & " +  str(int(allResults[bestId, EFFECTIVE_NUMBER_OF_COVARIATES_ID])) + " & " + str(allResults[bestId, criteriaID]))
    print("RESULT WITH BEST ANMI = " + str(allResults[bestId, constants.ANMI_ID]) + " & " +  str(int(allResults[bestId, constants.EFFECTIVE_NUMBER_OF_COVARIATES_ID])) + " & " + str(allResults[bestId, criteriaID]))
    helper.showVecInt(numpy.asarray(allClusterings[bestId]))
    bestClusteringFound = helper.getClusterIds(allClusterings[bestId])
    print("bestClusteringFound = ", bestClusteringFound)
    print("trueClusterIds = ", trueClusterIds)
    adjustedNMI = sklearn.metrics.adjusted_mutual_info_score(trueClusterIds, bestClusteringFound)
    print("adjustedNMI = ", adjustedNMI)
    return


    
        
    

def showResults(testSetting, NUMBER_OF_SAMPLES_PER_CLASS, METHOD):
    assert(METHOD != "onlyGamma")
    
    TOTAL_NUMBER_OF_FOLDS = 10
    
    DATA_SPECIFIER_STRING = testSetting + "_" + str(NUMBER_OF_SAMPLES_PER_CLASS) + "sc"
    covariateSims, dataFeature_allFolds, dataLabels_allFolds, trueClusterIds, trueRelevantCovariates, NUMBER_OF_CLASSES, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS = syntheticDataGeneration.generateData(testSetting, NUMBER_OF_SAMPLES_PER_CLASS, TOTAL_NUMBER_OF_FOLDS)
     
    print("trueClusterIds = ")
    print(trueClusterIds)
    
    avgNeighbours = "all"
       
    
    DATA_NAME ="SYNTHETIC_DATA"
    TRAIN_DATA_SEPCIFIER = testSetting + "_" + str(NUMBER_OF_SAMPLES_PER_CLASS) + "sc"
    
     
    if METHOD == "onlyNu":
        outputForLatex = "\multirowcell{5}{+ CLAW \\\\ Clustering} "
    elif METHOD == "kMeansClustering":
        outputForLatex = "\multirowcell{5}{+ k-means \\\\ Clustering} "
    elif METHOD == "convexClustering":
        outputForLatex = "\multirowcell{5}{+ Convex \\\\ Clustering} "
    else:
        assert(False)
    
    
    allANMIs_marginalLikelihood = numpy.zeros(TOTAL_NUMBER_OF_FOLDS)
    allANMIs_trainCVlogProb = numpy.zeros(TOTAL_NUMBER_OF_FOLDS)
    allANMIs_trainCVacc = numpy.zeros(TOTAL_NUMBER_OF_FOLDS)
    allANMIs_oracle = numpy.zeros(TOTAL_NUMBER_OF_FOLDS)
    
    for foldId in range(TOTAL_NUMBER_OF_FOLDS):
    
        print("**************** RESULTS FOR FOLD " + str(foldId) + " ********************************")
        
        filename = helper.EVALUATION_RESULTS_FOLDER + DATA_NAME + TRAIN_DATA_SEPCIFIER + "_" + METHOD + "_" + str(avgNeighbours) + "Neighbours_" + str(foldId) + "fold"
        allResults = numpy.load(filename + ".npy")
        
        allClusterings = None
        with open(filename + "_clusterings.pkl", "rb") as f:
            allClusterings = pickle.load(f)
       
        
        allANMIs_marginalLikelihood[foldId] = getANMI(allResults, constants.LOG_MARGINAL_LAPLACE_DIAG_VALIDATION_CRITERIA_ID)
        allANMIs_trainCVlogProb[foldId] = getANMI(allResults, constants.TRAIN_CV_LOG_PROB_ID)
        allANMIs_trainCVacc[foldId] = getANMI(allResults, constants.TRAIN_CV_ACC_ID)
        allANMIs_oracle[foldId] = getANMI(allResults, constants.ANMI_ID)
                
        print("LOG_MARGINAL_LAPLACE_DIAG_VALIDATION_CRITERIA_ID")
        showDetails(trueClusterIds, allClusterings, allResults, constants.LOG_MARGINAL_LAPLACE_DIAG_VALIDATION_CRITERIA_ID)
        
        print("TRAIN_CV_ACC_ID")
        showDetails(trueClusterIds, allClusterings, allResults, constants.TRAIN_CV_ACC_ID)
         
        print("TOTAL NUMBER OF CLUSTERING RESULTS = ", allResults.shape[0])
        
    print(METHOD)
    print(testSetting + " " + str(NUMBER_OF_SAMPLES_PER_CLASS))
    print("AVERAGE ANMI SELECED WITH MARGINAL LIKELIHOOD  = ", helper.showAvgAndStd(allANMIs_marginalLikelihood, digits = 2))
    print("AVERAGE ANMI SELECED WITH TRAIN-CV-LOGPROB = ", helper.showAvgAndStd(allANMIs_trainCVlogProb, digits = 2))
    print("AVERAGE ANMI SELECED WITH TRAIN-CV-ACC = ", helper.showAvgAndStd(allANMIs_trainCVacc, digits = 2))
    print("AVERAGE ANMI ORACLE PERFORMANCE = ", helper.showAvgAndStd(allANMIs_oracle, digits = 2))







