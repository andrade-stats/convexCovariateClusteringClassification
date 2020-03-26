import numpy
import helper
from graphviz import Digraph
from collections import defaultdict

def showShowOneClusterText(dimToWord, cluster):
    clusterRep = "{ "
    clusterWords = [dimToWord[elemId] for elemId in cluster]
    clusterRep += ", ".join(clusterWords)
    clusterRep += "}"
    print(clusterRep)
    

def getHashRep(cluster):
    clusterList = list(cluster)
    clusterList = numpy.sort(clusterList)
    clusterListAsStr = [str(elem) for elem in clusterList]
    return " ".join(clusterListAsStr)


def getSetRep(clusterAsStr):
    assert(isinstance(clusterAsStr, str))
    return set([int(elem) for elem in clusterAsStr.split(" ")])

def showCollectedInfo(dimToWord, allClassLabels, collectedInfo):
    
    classToClusters = []
    for i in range(len(allClassLabels)):
        classToClusters.append([])
    
    for clusterAsStr in collectedInfo.keys():
        allOccurrences = collectedInfo[clusterAsStr]
        allClassIds = numpy.ones(len(allOccurrences), dtype = numpy.int) * -1
        allOddsRatios = numpy.ones(len(allOccurrences)) * -1
        for i, occurrence in enumerate(allOccurrences):
            allClassIds[i], allOddsRatios[i] = occurrence
         
        elemsInClusterSet = getSetRep(clusterAsStr) 
         
        if not numpy.all(allClassIds == allClassIds[0]):
            print("classes = ", [allClassLabels[classId] for classId in allClassIds])
            # print("allClassIds = ", allClassIds)
            print("allOddsRatios = ", allOddsRatios)
            showShowOneClusterText(dimToWord, elemsInClusterSet)
        else:
            classId = allClassIds[0]
            assert(numpy.all(allClassIds == classId)) # all class ids should be the same
            classToClusters[classId].append(clusterAsStr)
            # print("allClassIds = ", allClassIds)
            # print("allOddsRatios = ", allOddsRatios)
        # assert(False)
    
    return classToClusters


# reading checked    
def getClusterScore(collectedInfo, clusterAsStr):
    assert(isinstance(clusterAsStr, str))
    allOccurrences = collectedInfo[clusterAsStr]
    return len(allOccurrences)
    
# reading checked
def getConflictingClusters(nonConflictingClusters, queryClusterAsStr):
    queryCluster = getSetRep(queryClusterAsStr)
    allConflictingClusters = set()
    for registeredClusterAsStr in nonConflictingClusters:
        registeredCluster = getSetRep(registeredClusterAsStr)
        assert(registeredCluster != queryCluster)
        if len(registeredCluster & queryCluster) >= 1 and len(registeredCluster - queryCluster) >= 1 and len(queryCluster - registeredCluster) >= 1:
            allConflictingClusters.add(registeredClusterAsStr)  

    return allConflictingClusters


# reading checked
def filterConflictingClusters(collectedInfo, allClustersFromOneClass):
    
    nonConflictingClusters = set()
    
    for candidateCluster in allClustersFromOneClass:
        candidateScore = getClusterScore(collectedInfo, candidateCluster)
        betterThanCurrent = True
        allConflictingClusters = getConflictingClusters(nonConflictingClusters, candidateCluster)
        for confCluster in allConflictingClusters:
            if getClusterScore(collectedInfo, confCluster) >= candidateScore:
                betterThanCurrent = False
                break
        if betterThanCurrent:
            
            nonConflictingClusters = nonConflictingClusters - allConflictingClusters
            
            if len(allConflictingClusters) > 0:
                print("****")
                print("candidateCluster = ", candidateCluster)
                print("candidateScore = ", candidateScore)
                print("nonConflictingClusters = ", nonConflictingClusters)
                print("allConflictingClusters = ", allConflictingClusters)
                allScores = [str(getClusterScore(collectedInfo, confCluster)) for confCluster in allConflictingClusters]
                print(",".join(allScores))
            # print("nonConflictingClusters = ", nonConflictingClusters)
            # assert(nonConflictingClusters.issubset(allConflictingClusters))
            nonConflictingClusters.add(candidateCluster)
    
    return nonConflictingClusters
    

   
        



def convertToWindowsFriendlyName(strName):
    niceStr = strName.replace(" ", "_")
    niceStr = niceStr.replace(".", "_")
    niceStr = niceStr.replace("-", "_")
    return niceStr


def getOddsRatios(classWeights, classLabels):
    assert(classWeights.shape[0] == len(classLabels))
        
    sortedClassWeights = - numpy.sort(- classWeights, axis = 0)
    bestClassMatches = numpy.argmax(classWeights, axis = 0)
    oddsRatio = numpy.exp(sortedClassWeights[0] - sortedClassWeights[1])
    
    assert(numpy.all(oddsRatio >= 1.0))
    return oddsRatio, bestClassMatches






class TreeNode:
    
    # checked
    def __init__(self, nodeContent, motherNode, childrenNodes, associatedClass, associatedOddsRatio):
        assert(isinstance(nodeContent, set))
        
        self.content = nodeContent   
        self.motherNode = motherNode
        self.associatedClass = associatedClass
        self.associatedOddsRatio = associatedOddsRatio
        
        if childrenNodes is not None:
            assert(len(self.content) > 1)
            self.childrenNodes = list(childrenNodes)
              
        else:
            # this can only be if we are at leaf
            assert(len(self.content) == 1)
            self.childrenNodes = None
        
        return
     
    def isLeaf(self):
        return (self.childrenNodes is None)
        
    def getLeaf(self):
        assert(self.isLeaf())
        assert(len(self.content) == 1)
        return (list(self.content))[0]
     
    # checked
    def updateChildren(self, newChildNode):
        
        newChildrenList = []
        for oldChildNode in self.childrenNodes:
            if not oldChildNode.content.issubset(newChildNode.content):
                assert(len(oldChildNode.content & newChildNode.content) == 0)
                newChildrenList.append(oldChildNode)
                 
        newChildrenList.append(newChildNode)
        self.childrenNodes = newChildrenList
                
        return


# checked
def getParentRootNode(allRootNodes, cluster):
    assert(isinstance(cluster, set))
    
    for rootNode in allRootNodes:
        if cluster == rootNode.content:
            return None
        elif cluster.issubset(rootNode.content):
            return rootNode
    
    return None


        
   
# checked
# sortedIds defines the order from root to leave, i.e. the clustering at sortedIds[0] is the the root (more precisely one root for one clustering)  
# requires that at least one partition is the partition with number of clusters = number of covariates
def getHierarchicalStructure(sortedIds, allClusterings, allOddsRatios, allBestClassLabels):
    assert(len(sortedIds) == len(allClusterings))
    assert(len(allClusterings) == len(allOddsRatios))
    assert(len(allClusterings) == len(allBestClassLabels))
    
    # numberOfRootNodesPerClass = defaultdict(int)
    
    # best clustering result is set as the root clustering
    bestClusteringResult = allClusterings[sortedIds[0]]
    oddsRatio, bestClassMatches = allOddsRatios[sortedIds[0]], allBestClassLabels[sortedIds[0]]
    assert(len(oddsRatio) == len(bestClassMatches))
    allRootNodes = []
    for clusterId, cluster in enumerate(bestClusteringResult):
        
        if len(cluster) == 1:
            # create a leaf node
            newTreeNode = TreeNode(cluster, None, None, bestClassMatches[clusterId], oddsRatio[clusterId])
        else:
            newTreeNode = TreeNode(cluster, None, [], bestClassMatches[clusterId], oddsRatio[clusterId])
            
        allRootNodes.append(newTreeNode)
    
    
    for i in sortedIds[1 : len(sortedIds)]:
        oddsRatio, bestClassMatches = allOddsRatios[i], allBestClassLabels[i]
        assert(len(oddsRatio) == len(bestClassMatches))
        for clusterId, cluster in enumerate(allClusterings[i]):
            
            # print("try to add cluster ", cluster)
            rootNode = getParentRootNode(allRootNodes, cluster)
            if rootNode is not None:
                # print("add to ", rootNode)
                addToTreeIfFitting(rootNode, cluster, bestClassMatches[clusterId], oddsRatio[clusterId])
        

    # safety check
    for rootNode in allRootNodes:
        checkFullUniqueContainment(rootNode)

    return allRootNodes





# is only used for safety check, i.e. to check that implementation is correct
def checkFullUniqueContainment(node):
    
    if node.childrenNodes is None:
        # we have a leaf
        return
    else:
        testSet = node.content
        assert(len(testSet) >= 1)
        assert(len(node.childrenNodes) >= 1)
        allCovariateIdsInChildren = set()
        
        for childNode in node.childrenNodes:
            assert(len(allCovariateIdsInChildren & childNode.content) == 0)
            allCovariateIdsInChildren = allCovariateIdsInChildren | childNode.content
    
        assert(len(allCovariateIdsInChildren & testSet) == len(testSet))
        
        for childNode in node.childrenNodes:
            checkFullUniqueContainment(childNode)


    
# extracted with help of http://tools.medialab.sciences-po.fr/iwanthue/
ALL_COLORS = ["#8e9524",
"#8a4be4",
"#4ca938",
"#ca4dd3",
"#587e37",
"#5f32a2",
"#d8851e",
"#6677de",
"#e24720",
"#3c9470",
"#d443a6",
"#9e7e36",
"#a263b6",
"#bd622d",
"#607bb6",
"#d13f3c",
"#a7507f",
"#9e5e3a",
"#db3e6e",
"#ac5055"]


def showTreeAsGraph(allRootNodes, dimToWord, allClassLabels, showClassId, outputFilename):
    
    totalNumberOfCovariatesInClass = 0
    # get all nodes belonging to class "showClassId"
    rootNodesInClass = []
    for rootNode in allRootNodes:
        if rootNode.associatedClass == showClassId:
            rootNodesInClass.append(rootNode)
            
            assert(len(rootNode.content) >= 1)
            totalNumberOfCovariatesInClass += len(rootNode.content) 
    
    
    print("ALL TREES IN CLASS AS TEXT:")
    showTreeAsText(rootNodesInClass, dimToWord, allClassLabels)
    
    assert(len(rootNodesInClass) >= 1)
    
    
    print("START VISUALIZATION:")
    # wholeGraph = Digraph("Covariate Cluster Visualization", edge_attr = {"dir" : "none"}, graph_attr = {"splines" : "ortho", "rankdir" : "LR", "concentrate" : "true"})
    wholeGraph = Digraph("Covariate Cluster Visualization", edge_attr = {"dir" : "none"}, graph_attr = {"splines" : "line", "rankdir" : "LR", "concentrate" : "true"})
    
    wholeGraph.node_attr.update(shape = "plain", style='rounded', color=ALL_COLORS[showClassId], fontcolor=ALL_COLORS[showClassId])
    
    drawAllNodesOnSameLevel(wholeGraph, dimToWord, 0, {}, None, rootNodesInClass)
    
    print("totalNumberOfCovariatesInClass = ", totalNumberOfCovariatesInClass)
    print("START RENDERING:")
    wholeGraph.render(helper.PLOT_RESULTS_FOLDER + outputFilename, view=False, cleanup = True)

    
    return



def drawAllNodesOnSameLevel(wholeGraph, dimToWord, levelId, levelIdToNodeId, parentNodeName, allNodesThisLevel):
        
    # this ensures that nodes with high odds ratio are listed top (order of sorting has to be changed depending if root or not because of bug in graphviz?)    
    if parentNodeName is None:
        allNodesThisLevel = sorted(allNodesThisLevel, key=lambda node: node.associatedOddsRatio)
    else:
        allNodesThisLevel = sorted(allNodesThisLevel, key=lambda node: -node.associatedOddsRatio)
     
     
    levelId = levelId + 1
    if not (levelId in levelIdToNodeId):
        levelIdToNodeId[levelId] = 0
        
    for currentNode in allNodesThisLevel:
        levelIdToNodeId[levelId] = levelIdToNodeId[levelId] + 1 
        currentNodeName = str(levelId) + "-" + str(levelIdToNodeId[levelId])
       
        # draw node
        if currentNode.isLeaf():
            nodeText = dimToWord[currentNode.getLeaf()]
            # print("nodeText = ", nodeText)
            # if "&" in nodeText:
            nodeText = nodeText.replace("<","&lt;")
            nodeText = nodeText.replace(">","&gt;")
            nodeText = nodeText.replace("&","&amp;")
            
        else:
            nodeText = "<i>(size " + str(len(currentNode.content)) + ")</i>"
       
        weightAsStr = str(round(currentNode.associatedOddsRatio,2))
        wholeGraph.node(currentNodeName, label = '''<<table border="0" cellborder="0" cellspacing="0" color="white">
                    <tr><td port="1"><font color="black">''' + nodeText + '''</font></td></tr>
                    <tr><td port="2" border="1">''' + weightAsStr + '''</td></tr>
                </table>>''', color=ALL_COLORS[currentNode.associatedClass], fontcolor=ALL_COLORS[currentNode.associatedClass])
        
        # draw edge to parent
        if parentNodeName is not None:
            # print("draw edge: " + parentNodeName + " <-> " + currentNodeName)
            wholeGraph.edge(parentNodeName, currentNodeName)
    
        # draw its children
        if not currentNode.isLeaf():
            drawAllNodesOnSameLevel(wholeGraph, dimToWord, levelId, levelIdToNodeId, currentNodeName, currentNode.childrenNodes)
    
    return


def drawColorLegend(DATA_NAME, allClassLabels):
    wholeGraph = Digraph("LEGEND FOR " + DATA_NAME, edge_attr = {"dir" : "none"}, graph_attr = {"splines" : "ortho", "rankdir" : "LR", "concentrate" : "true"})
    wholeGraph.node_attr.update(shape = "record", style='rounded')
    
    for i in range(len(allClassLabels)-1, -1, -1):
        wholeGraph.node('label' + str(i), label = allClassLabels[i], color=ALL_COLORS[i], fontcolor=ALL_COLORS[i])
    
    wholeGraph.render(helper.PLOT_RESULTS_FOLDER + DATA_NAME + "_legend", view=False, cleanup = True)


# checked
def addToTreeIfFitting(startNode, queryCluster, queryClustersClass, queryClustersOddsRatio):
    assert(queryCluster.issubset(startNode.content) and len(queryCluster) < len(startNode.content))
    
    # print("-------------------------------------")
    childrenNodesOfQueryCluster = []
    for childNode in startNode.childrenNodes:
        # print("check child node ", childNode.content)
        if queryCluster == childNode.content:
            # node already exists
            return None
        elif queryCluster.issubset(childNode.content):
            # query cluster is true subset of child
            return addToTreeIfFitting(childNode, queryCluster, queryClustersClass, queryClustersOddsRatio)
        elif childNode.content.issubset(queryCluster):
            # all of childNode is contained in queryCluster
            childrenNodesOfQueryCluster.append(childNode)
        elif len(childNode.content & queryCluster) >= 1:
            # clusters are partly overlapping
            return None
        # otherwise, the clusters are non-overlapping -> nothing to do
    
    if len(queryCluster) == 1:
        # newNode is a leaf
        assert(len(childrenNodesOfQueryCluster) == 0)
        newNode = TreeNode(queryCluster, startNode, None, queryClustersClass, queryClustersOddsRatio)
    else:
        newNode = TreeNode(queryCluster, startNode, childrenNodesOfQueryCluster, queryClustersClass, queryClustersOddsRatio)
        
    startNode.updateChildren(newNode)
    return 


def getFinestClusteringId(allClusterings):
    allIds = set()
    for cluster in allClusterings[0]:
        allIds = allIds | cluster
        
    NUMBER_OF_COVARIATES = len(allIds)
    
    finestClusteringId = None
    for clusteringId, clustering in enumerate(allClusterings):
        if len(clustering) == NUMBER_OF_COVARIATES:
            finestClusteringId = clusteringId
            break
    
    assert(finestClusteringId is not None)
    return finestClusteringId


# allRootNodes = visualizer.getHierarchicalStructure(idsForSorting, allClusterings, allOddsRatios, allBestClassLabels)

# def getFlatStructure(selectedClusteringId, allClusterings, allClustersClassWeights, allClassLabels, wordToDim, dimToWord):
def getFlatStructure(selectedClusteringId, allClusterings, allOddsRatios, allBestClassLabels):
    
    finestClusteringId = getFinestClusteringId(allClusterings)
       
    
    selectedClusterings = [allClusterings[selectedClusteringId], allClusterings[finestClusteringId]]
    selectedOddsRatios = [allOddsRatios[selectedClusteringId], allOddsRatios[finestClusteringId]]
    selectedBestClassLabels = [allBestClassLabels[selectedClusteringId], allBestClassLabels[finestClusteringId]]
    return getHierarchicalStructure([0,1], selectedClusterings, selectedOddsRatios, selectedBestClassLabels)


def getLowRelevanceCovariates(allClusterings, allClustersClassWeights, allClassLabels, VISUALIZATION_ODDS_RATIO_THRESHOLD):
    finestClusteringId = getFinestClusteringId(allClusterings)
    oddsRatio, bestClassMatches = getOddsRatios(allClustersClassWeights[finestClusteringId], allClassLabels)
    assert(len(oddsRatio) == len(bestClassMatches))
    
    allFilteredOutCovariates = set()
    for clusterId, singleCluster in enumerate(allClusterings[finestClusteringId]):
        assert(len(singleCluster) == 1)
        if oddsRatio[clusterId] < VISUALIZATION_ODDS_RATIO_THRESHOLD:
            allFilteredOutCovariates = allFilteredOutCovariates | singleCluster
    
    return allFilteredOutCovariates


# for large trees: filter out covariates with small odds ratio:
def filterOutCovariates(allClusterings, allClustersClassWeights, allClassLabels, VISUALIZATION_ODDS_RATIO_THRESHOLD):
    
    lowRelevanceCovariates = getLowRelevanceCovariates(allClusterings, allClustersClassWeights, allClassLabels, VISUALIZATION_ODDS_RATIO_THRESHOLD)
    
    allClusterings_updated = []
    allOddsRatios_updated = []
    allBestClassMatches_updated = []
    for clusteringResultId, fullClustering in enumerate(allClusterings):
        assert(len(fullClustering) == (allClustersClassWeights[clusteringResultId]).shape[1])
        oddsRatios, bestClassMatches = getOddsRatios(allClustersClassWeights[clusteringResultId], allClassLabels)
        assert(oddsRatios.shape[0] == len(fullClustering))
        assert(bestClassMatches.shape[0] == len(fullClustering))
        
        fullClustering_updated = []
        oddsRatios_updated = []
        bestClassMatches_updated = []
        for clusterId, cluster in enumerate(fullClustering):
            filterOutIds = lowRelevanceCovariates & cluster
            newCluster = cluster - filterOutIds
            if len(newCluster) > 0:
                fullClustering_updated.append(newCluster)
                oddsRatios_updated.append(oddsRatios[clusterId])
                bestClassMatches_updated.append(bestClassMatches[clusterId])
        
        allClusterings_updated.append(fullClustering_updated)
        allOddsRatios_updated.append(numpy.asarray(oddsRatios_updated))
        allBestClassMatches_updated.append(numpy.asarray(bestClassMatches_updated))
        
    
    assert(len(allClusterings_updated) == len(allClusterings))
    
    print("REDUCED NUMBER OF COVARIATES By ", len(lowRelevanceCovariates))
    return allClusterings_updated, allOddsRatios_updated, allBestClassMatches_updated
    

def showTreeAsText(allRootNodes, dimToWord, allClassLabels, offset = ""):
    
    level = len(offset)
    for rootNode in allRootNodes:
        wordSet = set([dimToWord[wordId] for wordId in rootNode.content])
        classAssociationInfo = "associated class = " + allClassLabels[rootNode.associatedClass] + ", odds ratio = " + str(rootNode.associatedOddsRatio)
        print(offset + "Level " + str(level) + ": " + str(wordSet) + " | " + classAssociationInfo)
        if rootNode.childrenNodes is not None:
            showTreeAsText(rootNode.childrenNodes, dimToWord, allClassLabels, offset + "\t")
    
    return
