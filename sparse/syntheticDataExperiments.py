
import sys
import numpy
import syntheticDataGeneration
import experiments
import helper
import cvxpyTraining


# Examples:
# /opt/intel/intelpython3/bin/python syntheticDataExperiments.py smallContra 1000 cvxpy
# /opt/intel/intelpython3/bin/python syntheticDataExperiments.py smallContra 100 proposed

def main():

    testSetting = sys.argv[1]
    NUMBER_OF_SAMPLES_PER_CLASS = int(sys.argv[2])
    hyperparametersRange = "onlyNu"
    
    optParams = {}
    if sys.argv[3] == "cvxpy":
        optParams["CVXPY_USED"] = "_trainedWithCVXPY"
        optParams["SOLVER"] = sys.argv[4] 
    elif sys.argv[3] == "proposed":
        optParams["CVXPY_USED"] = ""
        optParams["SOLVER"] = ""
    else:
        assert(False)
    
    assert(NUMBER_OF_SAMPLES_PER_CLASS >= 10)
    
    TOTAL_NUMBER_OF_FOLDS = 10
    
    allNus, allGammas = helper.getHyperparameters(hyperparametersRange)
    
    optParams["AVG_NEIGHBOURS"] = "all"
    optParams["INITIAL_RHO"] = 1.0
    optParams["MAX_LBFGS_ITERATIONS"] = 15000 # default value
    optParams["ADMM_MAX_ITERATIONS"] = 10000 
      
    optParams["RHO_MULTIPLIER"] = 1.0
    optParams["EPSILON"] = 0.00001

    covariateSims, dataFeature_allFolds, dataLabels_allFolds, trueClusterIds, trueRelevantCovariates, NUMBER_OF_CLASSES, NUMBER_OF_LATENT_CLUSTERS, NUMBER_OF_COVARIATES_PER_CLUSTER, IRRELEVANT_CLUSTERS, CONTRADICTING_CLUSTERS = syntheticDataGeneration.generateData(testSetting, NUMBER_OF_SAMPLES_PER_CLASS, TOTAL_NUMBER_OF_FOLDS)
    
    print("initial rho = ", optParams["INITIAL_RHO"])
    print("rho multiplier = ", optParams["RHO_MULTIPLIER"])
    print("EPSILON = ", optParams["EPSILON"])
    print("ADMM_MAX_ITERATIONS = ", optParams["ADMM_MAX_ITERATIONS"])
   
    allRuntimes = numpy.zeros(TOTAL_NUMBER_OF_FOLDS)
    
    for foldId in range(TOTAL_NUMBER_OF_FOLDS):
        DATA_SPECIFIER_STRING = testSetting + "_" + str(NUMBER_OF_SAMPLES_PER_CLASS) + "sc" + "_" + str(foldId)
        BASE_FILENAME = helper.getBaseFilename("SYNTHETIC_DATA", DATA_SPECIFIER_STRING)
        
        if optParams["CVXPY_USED"] == "":
            # use the prospoed fast ADMM method
            allRuntimes[foldId] = experiments.runTraining(BASE_FILENAME, dataFeature_allFolds[foldId], dataLabels_allFolds[foldId], covariateSims, allNus, allGammas, optParams)
        elif optParams["CVXPY_USED"] == "_trainedWithCVXPY":
            # use the standard solver from cvxpy
            allRuntimes[foldId]= cvxpyTraining.proposedMethodTrainedWithCVXPY(BASE_FILENAME, dataFeature_allFolds[foldId], dataLabels_allFolds[foldId], covariateSims, allNus, allGammas, optParams)
        else:
            assert(False)
      
    
    # transform into minutes
    allRuntimes = allRuntimes / 60.0 
    
    print("-----------------")
    print(testSetting)
    print(NUMBER_OF_SAMPLES_PER_CLASS)
    if optParams["CVXPY_USED"] == "_trainedWithCVXPY": 
        print("CVXPY USED = ", optParams["CVXPY_USED"])
        print("SOLVER = ", optParams["SOLVER"])
    else:
        print("PROPOSED METHOD USED")
    print("average runtime in minutes = ", helper.showAvgAndStd(allRuntimes, digits = 3))
    print("-----------------")
    
    print("avgNeighbours = ", optParams["AVG_NEIGHBOURS"])
    print("initial rho = ", optParams["INITIAL_RHO"])
    print("rho multiplier = ", optParams["RHO_MULTIPLIER"])
    print("EPSILON = ", optParams["EPSILON"])
    print("ADMM_MAX_ITERATIONS = ", optParams["ADMM_MAX_ITERATIONS"])
    print("allNus = ", allNus)
    print("allGammas = ", allGammas)
    print("SYNTHETIC DATA")
    print("NUMBER_OF_SAMPLES_PER_CLASS = ", NUMBER_OF_SAMPLES_PER_CLASS)
    print("Finished Training Successfully")
    print("TOTAL_NUMBER_OF_FOLDS = ", TOTAL_NUMBER_OF_FOLDS)
    print("testSetting = ", testSetting)
    

main()
