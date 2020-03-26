
import time
import ADMM
import helper
import sklearn.model_selection
import multiprocessing
import numpy

def saveStatistics(BASE_FILENAME, optParams, nu, gamma, partialConnectedInfo, B, beta):
    
    OPT_PARAM_STRING = str(optParams["MAX_LBFGS_ITERATIONS"]) + "lbfgsMaxIt"
    OPT_PARAM_STRING += "_" + str(optParams["EPSILON"]) + "epsilon"
    OPT_PARAM_STRING += "_" + str(optParams["INITIAL_RHO"]) + "initRho"
    OPT_PARAM_STRING += "_" + str(optParams["RHO_MULTIPLIER"]) + "rhoMult"
    OPT_PARAM_STRING += "_" + str(optParams["AVG_NEIGHBOURS"]) + "avgNeighbours"
    OPT_PARAM_STRING += "_" + str(optParams["ADMM_MAX_ITERATIONS"]) + "admmMaxIt"
    OPT_PARAM_STRING += "_" + str(nu) + "nu_" + str(gamma) + "gamma" 
    OPT_PARAM_STRING += optParams["CVXPY_USED"] + optParams["SOLVER"]
    
    # clusterIds, relevance = fullyConnectedInfo
    # numpy.save(BASE_FILENAME + OPT_PARAM_STRING + "_clusterIdsFullyConnected", clusterIds)
    # numpy.save(BASE_FILENAME + OPT_PARAM_STRING + "_relevanceFullyConnected", relevance)
    
    clusterIds, relevance = partialConnectedInfo
    numpy.save(BASE_FILENAME + OPT_PARAM_STRING + "_clusterIdsPartialConnected", clusterIds)
    numpy.save(BASE_FILENAME + OPT_PARAM_STRING + "_relevancePartialConnected", relevance)
    
    numpy.save(BASE_FILENAME + OPT_PARAM_STRING + "_B", B)
    numpy.save(BASE_FILENAME + OPT_PARAM_STRING + "_beta", beta)
   
    print("Saved everything successfully")
    return


def loadStatistics(BASE_FILENAME, optParams, nu, gamma):
    OPT_PARAM_STRING = str(optParams["MAX_LBFGS_ITERATIONS"]) + "lbfgsMaxIt"
    OPT_PARAM_STRING += "_" + str(optParams["EPSILON"]) + "epsilon"
    OPT_PARAM_STRING += "_" + str(optParams["INITIAL_RHO"]) + "initRho"
    OPT_PARAM_STRING += "_" + str(optParams["RHO_MULTIPLIER"]) + "rhoMult"
    OPT_PARAM_STRING += "_" + str(optParams["AVG_NEIGHBOURS"]) + "avgNeighbours"
    OPT_PARAM_STRING += "_" + str(optParams["ADMM_MAX_ITERATIONS"]) + "admmMaxIt"
    OPT_PARAM_STRING += "_" + str(nu) + "nu_" + str(gamma) + "gamma" 
    OPT_PARAM_STRING += optParams["CVXPY_USED"] + optParams["SOLVER"]
    
    # allClusterIds = numpy.load(BASE_FILENAME + OPT_PARAM_STRING + "_clusterIdsFullyConnected" + ".npy")
    # allRelevanceIds = numpy.load(BASE_FILENAME + OPT_PARAM_STRING + "_relevanceFullyConnected" + ".npy")
    # fullConnectedInfo = (allClusterIds, allRelevanceIds)
    
    allClusterIds = numpy.load(BASE_FILENAME + OPT_PARAM_STRING + "_clusterIdsPartialConnected" + ".npy")
    allRelevanceIds = numpy.load(BASE_FILENAME + OPT_PARAM_STRING + "_relevancePartialConnected" + ".npy")
    partialConnectedInfo = (allClusterIds, allRelevanceIds)
    
    B = numpy.load(BASE_FILENAME + OPT_PARAM_STRING + "_B" + ".npy")
    beta = numpy.load(BASE_FILENAME + OPT_PARAM_STRING + "_beta" + ".npy")
    
    print("Loaded everything successfully")
    return partialConnectedInfo, B, beta


def runTraining(BASE_FILENAME, trainingData, trainingLabel, covariateSims, allNus, allGammas, optParams):
    assert(optParams["CVXPY_USED"] == "" and optParams["SOLVER"] == "")
    
    start_time_allExperiments = time.time()
    
    NUMBER_OF_PARAMETER_COMBINATIONS = len(allNus) * len(allGammas)
    notConvergedCount = 0
    
    assert(len(allGammas) == 1)
    allDurations = numpy.zeros(len(allNus))

    warmStartAuxilliaryVars = None
    
    paramId = 0
    for itNr, nu in enumerate(allNus):
        for gamma in allGammas:
            start_time_oneExperiment = time.time()

            paramId += 1
            B, beta, partialConnectedInfo, converged, warmStartAuxilliaryVars = ADMM.parallelADMM(trainingData, trainingLabel, covariateSims, nu, gamma, optParams, warmStartAuxilliaryVars, paramId)
              
            if not converged:
                notConvergedCount += 1
    
            allDurations[itNr] = (time.time() - start_time_oneExperiment) / 60.0

            if itNr > 5 and allDurations[itNr] > 20.0 and allDurations[itNr-1] > 20.0:
                print("--- ONE RUN IS EXPECTED TO TAKE MORE THAN 24h --- ")
                print("allDurations[itNr-1] = ", allDurations[itNr-1])
                print("allDurations[itNr] = ", allDurations[itNr])
                print("--- EXIT --- ")
                assert(False)
                
            saveStatistics(BASE_FILENAME, optParams, nu, gamma, partialConnectedInfo, B, beta)
            
            microScoreTrain, macroScoreTrain, accuracyTrain = ADMM.evaluate(trainingData, trainingLabel, B, beta)
            print("microScoreTrain, macroScoreTrain, accuracyTrain = ", (microScoreTrain, macroScoreTrain, accuracyTrain))
        
            clusterIds, relevance = partialConnectedInfo
            print("number of selected covariates = ", numpy.sum(relevance))
            print("clusterIds = ")
            helper.showVecInt(clusterIds)
    
    duration = (time.time() - start_time_allExperiments)
    
    print("-----------------")
    print("Finished successfully full training in (min) = ", round(duration / (60.0),3))
    print("-----------------")
    
    if (duration / 60.0) > 4500:
        print("STOP: ONE FULL RUN TAKES TOO LONG")
        assert(False)
    

    print("NUMBER_OF_PARAMETER_COMBINATIONS = ", NUMBER_OF_PARAMETER_COMBINATIONS)
    print("notConvergedCount = ", notConvergedCount)
    
    return duration


    
    