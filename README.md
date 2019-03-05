
-------------------------  Main implementation ---------------------------------
 
 ADMM.py
 parallelADMM(dataFeatures, dataLabels, covariateSims, origNu, origGamma, optParams, warmStartAuxilliaryVars, paramId)
 
 bStep.py
 optimizeLBFGSNew(dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B0, beta0, allAdjacentNodes, MAX_ITERATIONS)
 
 zStep.py
 updateZ_fast(edgesZ, singleZ, edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodes)
 
 uStep.py
 updateU_fast(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes)
 
 identify connected components:
 getClusteringAndSelectedFeatures(edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodesPrior)
 in zStep.py
 
------------------------- Experiments on synthetic data ---------------------------------

1. train with different hyperparameters, e.g.:
/opt/intel/intelpython3/bin/python syntheticDataExperiments.py smallContra 100 proposed

2. run all evaluation measures and save results, e.g.:
/opt/intel/intelpython3/bin/python evalAndSaveResults.py SYNTHETIC_DATA onlyNu 10 all smallContra 1000

3. visualize results:
/opt/intel/intelpython3/bin/python analyzeClustering.py
The results are save in the folder "plots".

------------------------------------ Visualization Details --------------------------------------------

the visualization in "analyzeClustering.py" uses the colors extracted from:
http://tools.medialab.sciences-po.fr/iwanthue/
and the graphviz package 
https://www.graphviz.org/

