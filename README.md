Convex Covariate Clustering for Classification
==

Python 3 Implementation of the method as proposed in 
http://arxiv.org/abs/1903.01680

 
Experiments on synthetic data
==

**1. Training**: Train with different hyperparameters, e.g.:

	```bash
 	$ python3 syntheticDataExperiments.py smallContra 100 proposed
	```

**2. Run clustering evaluation**: run all evaluation measures and save results, e.g.:

	```bash
	$ python3 evalAndSaveResults.py SYNTHETIC_DATA onlyNu 10 all smallContra 1000
	```

**3. Visualize Results**: The results are save in the folder "plots".

	```bash
	$ python3 analyzeClustering.py
	```


ADMM Implementation
==
 
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


Visualization Details
==

the visualization in "analyzeClustering.py" uses the colors extracted from:
http://tools.medialab.sciences-po.fr/iwanthue/
and the graphviz package 
https://www.graphviz.org/

