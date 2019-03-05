
import time
import sklearn.model_selection
import sklearn.metrics
import multiprocessing
import numpy

    
# checked
def eval(trueTestLabels, predictedTestLabels, evalCriteria):
    if evalCriteria == "micro":  
        return sklearn.metrics.f1_score(y_true = trueTestLabels, y_pred = predictedTestLabels, average='micro')
    elif evalCriteria == "macro":
        return sklearn.metrics.f1_score(y_true = trueTestLabels, y_pred = predictedTestLabels, average='macro')
    elif evalCriteria == "accuracy":
        return sklearn.metrics.accuracy_score(y_true = trueTestLabels, y_pred = predictedTestLabels)
    else:
        assert(False)
        
# checked
def runCV(allCovariatesAsMatrix, allLabels, trainAndTestFunc, allHyperparameters, evalCriteria, NUMBER_OF_FOLDS = 5):
        
    kfoldSplitter = sklearn.model_selection.StratifiedKFold(n_splits=NUMBER_OF_FOLDS, random_state=432532, shuffle=True)
    
    start_time_allExperiments = time.time()
    
    allResults = []
    allResultSD = []
    
    for hyperparams in allHyperparameters:
            
            allExperimentParams = []
            for train_index, test_index in kfoldSplitter.split(allCovariatesAsMatrix, allLabels):
                allParamsOneRun = allCovariatesAsMatrix[train_index], allLabels[train_index], allCovariatesAsMatrix[test_index], allLabels[test_index], hyperparams, evalCriteria
                allExperimentParams.append(allParamsOneRun) 
               
            myPool = multiprocessing.Pool(processes = NUMBER_OF_FOLDS)
            try:
                allResultsForOneHyperparam = myPool.map(trainAndTestFunc, allExperimentParams)
                myPool.close()
                myPool.join()
            except (KeyboardInterrupt):
                myPool.terminate()
                myPool.join()
                assert(False)
    
            meanValue = numpy.mean(numpy.asarray(allResultsForOneHyperparam))
            allResults.append(meanValue)
            allResultSD.append(numpy.std(numpy.asarray(allResultsForOneHyperparam)))


    duration = (time.time() - start_time_allExperiments)
    print("-----------------")
    print("Finished successfully all CV in (min) = ", round(duration / (60.0),3))
    print("-----------------")
    return numpy.asarray(allResults), numpy.asarray(allResultSD)
