#!/usr/bin/python
#
# The evaluation script for pixel-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below 
# and set the required environment variables as needed, such that 
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# Note that the script is a lot faster, if you enable cython support.
# WARNING: Cython only tested for Ubuntu 64bit OS.
# To enable cython, run
# setup.py build_ext --inplace
#
# To run this script, make sure that your results are images,
# where pixels encode the class IDs as defined in labels.py.
# Note that the regular ID is used, not the train ID.
# Further note that many classes are ignored from evaluation.
# Thus, authors are not expected to predict these classes and all
# pixels with a ground truth label that is ignored are ignored in
# evaluation.

# python imports
from __future__ import print_function
import os, sys
import platform
import fnmatch

try:
    from itertools import izip
except ImportError:
    izip = zip

# Kitti imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from csHelpers import *

# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False


###################################
# PLEASE READ THESE INSTRUCTIONS!!!
###################################
# Provide the prediction file for the given ground truth file.
#
# The current implementation expects the results to be in a certain root folder.
# This folder is one of the following with decreasing priority:
#   - environment variable KITTI_RESULTS
#   - environment variable KITTI_DATASET/results
#   - ../../results/"
#
# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
# <city>_123456_123456*.png
# for a ground truth filename
# <city>_123456_123456_gtFine_labelIds.png
def getPrediction( config, groundTruthFile ):
    # determine the prediction path, if the method is first called
    if not config.predictionPath:
        rootPath = None
        if 'KITTI_RESULTS' in os.environ:
            rootPath = os.environ['KITTI_RESULTS']
        elif 'KITTI_DATASET' in os.environ:
            rootPath = os.path.join( os.environ['KITTI_DATASET'] , "results" )
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root folder. Please read the instructions of this method.")

        config.predictionPath = rootPath

    # walk the prediction path, if not happened yet
    if not config.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(config.predictionPath):
            walk.append( (root,filenames) )
        config.predictionWalk = walk

    # csFile = getCsFileInfo(groundTruthFile)
    # filePattern = "{}_{}_{}*.png".format( csFile.city , csFile.sequenceNb , csFile.frameNb )
    filePattern = "{}.png".format(os.path.basename(groundTruthFile).split('.')[0])

    predictionFile = None
    for root, filenames in config.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    return predictionFile


######################
# Parameters
######################


# A dummy class to collect all bunch of data
class Config(object):
    pass
# And a global object of that class
config = Config()

# Where to look for Kitti
if 'KITTI_DATASET' in os.environ:
    config.kittiPath = os.environ['KITTI_DATASET']
else:
    config.kittiPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

# Parameters that should be modified by user
config.exportFile         = os.path.join( config.kittiPath , "evaluationResults" , "resultPixelLevelSemanticLabeling.json" )
config.groundTruthSearch  = os.path.join( config.kittiPath , "training" , "semantic", "*.png" )

# Remaining params
config.evalInstLevelScore = True
config.evalPixelAccuracy  = False
config.evalLabels         = []
config.printRow           = 5
config.normalized         = True
config.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
config.bold               = colors.BOLD if config.colorized else ""
config.nocol              = colors.ENDC if config.colorized else ""
config.JSONOutput         = True
config.quiet              = False

config.avgClassSize       = {
            "truck"     :  1801.103394,
            "person"    :  1234.262573,
            "train"     : 18448.460938,
            "bicycle"   :  1244.791260,
            "motorcycle":   695.617676,
            "bus"       :  2761.151611,
            "caravan"   :  1017.777771,
            "rider"     :  1318.393921,
            "trailer"   : 14457.615234,
            "car"       :  3669.255859,
}

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
config.predictionPath = None
config.predictionWalk = None


#########################
# Methods
#########################


# Generate empty confusion matrix and create list of relevant labels
def generateMatrix(config):
    config.evalLabels = []
    for label in labels:
        if (label.id < 0):
            continue
        # we append all found labels, regardless of being ignored
        config.evalLabels.append(label.id)
    maxId = max(config.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)

def generateInstanceStats(config):
    instanceStats = {}
    instanceStats["classes"   ] = {}
    instanceStats["categories"] = {}
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            instanceStats["classes"][label.name] = {}
            instanceStats["classes"][label.name]["tp"] = 0.0
            instanceStats["classes"][label.name]["tpWeighted"] = 0.0
            instanceStats["classes"][label.name]["fn"] = 0.0
            instanceStats["classes"][label.name]["fnWeighted"] = 0.0
    for category in category2labels:
        labelIds = []
        allInstances = True
        for label in category2labels[category]:
            if label.id < 0:
                continue
            if not label.hasInstances:
                allInstances = False
                break
            labelIds.append(label.id)
        if not allInstances:
            continue

        instanceStats["categories"][category] = {}
        instanceStats["categories"][category]["tp"] = 0.0
        instanceStats["categories"][category]["tpWeighted"] = 0.0
        instanceStats["categories"][category]["fn"] = 0.0
        instanceStats["categories"][category]["fnWeighted"] = 0.0
        instanceStats["categories"][category]["labelIds"] = labelIds

    return instanceStats


# Get absolute or normalized value from field in confusion matrix.
def getMatrixFieldValue(confMatrix, i, j, config):
    if config.normalized:
        rowSum = confMatrix[i].sum()
        if (rowSum == 0):
            return float('nan')
        return float(confMatrix[i][j]) / rowSum
    else:
        return confMatrix[i][j]

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, confMatrix, config):
    if id2label[label].ignoreInEval:
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in config.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular label
def getInstanceIouScoreForLabel(label, confMatrix, instStats, config):
    if id2label[label].ignoreInEval:
        return float('nan')

    labelName = id2label[label].name
    if not labelName in instStats["classes"]:
        return float('nan')

    tp = instStats["classes"][labelName]["tpWeighted"]
    fn = instStats["classes"][labelName]["fnWeighted"]
    # false postives computed as above
    notIgnored = [l for l in config.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate prior for a particular class id.
def getPrior(label, confMatrix):
    return float(confMatrix[label,:].sum()) / confMatrix.sum()

# Get average of scores.
# Only computes the average over valid entries.
def getScoreAverage(scoreList, config):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(scoreList[score]):
            validScores += 1
            scoreSum += scoreList[score]
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores

# Calculate and return IOU score for a particular category
def getIouScoreForCategory(category, confMatrix, config):
    # All labels in this category
    labels = category2labels[category]
    # The IDs of all valid labels in this category
    labelIds = [label.id for label in labels if not label.ignoreInEval and label.id in config.evalLabels]
    # If there are no valid labels, then return NaN
    if not labelIds:
        return float('nan')

    # the number of true positive pixels for this category
    # this is the sum of all entries in the confusion matrix
    # where row and column belong to a label ID of this category
    tp = np.longlong(confMatrix[labelIds,:][:,labelIds].sum())

    # the number of false negative pixels for this category
    # that is the sum of all rows of labels within this category
    # minus the number of true positive pixels
    fn = np.longlong(confMatrix[labelIds,:].sum()) - tp

    # the number of false positive pixels for this category
    # we count the column sum of all labels within this category
    # while skipping the rows of ignored labels and of labels within this category
    notIgnoredAndNotInCategory = [l for l in config.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular category
def getInstanceIouScoreForCategory(category, confMatrix, instStats, config):
    if not category in instStats["categories"]:
        return float('nan')
    labelIds = instStats["categories"][category]["labelIds"]

    tp = instStats["categories"][category]["tpWeighted"]
    fn = instStats["categories"][category]["fnWeighted"]

    # the number of false positive pixels for this category
    # same as above
    notIgnoredAndNotInCategory = [l for l in config.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom


# create a dictionary containing all relevant results
def createResultDict( confMatrix, classScores, classInstScores, categoryScores, categoryInstScores, perImageStats, config ):
    # write JSON result file
    wholeData = {}
    wholeData["confMatrix"] = confMatrix.tolist()
    wholeData["priors"] = {}
    wholeData["labels"] = {}
    for label in config.evalLabels:
        wholeData["priors"][id2label[label].name] = getPrior(label, confMatrix)
        wholeData["labels"][id2label[label].name] = label
    wholeData["classScores"] = classScores
    wholeData["classInstScores"] = classInstScores
    wholeData["categoryScores"] = categoryScores
    wholeData["categoryInstScores"] = categoryInstScores
    wholeData["averageScoreClasses"] = getScoreAverage(classScores, config)
    wholeData["averageScoreInstClasses"] = getScoreAverage(classInstScores, config)
    wholeData["averageScoreCategories"] = getScoreAverage(categoryScores, config)
    wholeData["averageScoreInstCategories"] = getScoreAverage(categoryInstScores, config)

    if perImageStats:
        wholeData["perImageScores"] = perImageStats

    return wholeData

def writeJSONFile(wholeData, config):
    path = os.path.dirname(config.exportFile)
    ensurePath(path)
    writeDict2JSON(wholeData, config.exportFile)

# Print confusion matrix
def printConfMatrix(confMatrix, config):
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in config.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=config.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=config.printRow + 3, fill='-', text=" "))

    # print label names
    print("\b{text:>{width}} |".format(width=13, text=""), end=' ')
    for label in config.evalLabels:
        print("\b{text:^{width}} |".format(width=config.printRow, text=id2label[label].name[0]), end=' ')
    print("\b{text:>{width}} |".format(width=6, text="Prior"))

    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in config.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=config.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=config.printRow + 3, fill='-', text=" "))

    # print matrix
    for x in range(0, confMatrix.shape[0]):
        if (not x in config.evalLabels):
            continue
        # get prior of this label
        prior = getPrior(x, confMatrix)
        # skip if label does not exist in ground truth
        if prior < 1e-9:
            continue

        # print name
        name = id2label[x].name
        if len(name) > 13:
            name = name[:13]
        print("\b{text:>{width}} |".format(width=13,text=name), end=' ')
        # print matrix content
        for y in range(0, len(confMatrix[x])):
            if (not y in config.evalLabels):
                continue
            matrixFieldValue = getMatrixFieldValue(confMatrix, x, y, config)
            print(getColorEntry(matrixFieldValue, config) + "\b{text:>{width}.2f}  ".format(width=config.printRow, text=matrixFieldValue) + config.nocol, end=' ')
        # print prior
        print(getColorEntry(prior, config) + "\b{text:>{width}.4f} ".format(width=6, text=prior) + config.nocol)
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in config.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=config.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=config.printRow + 3, fill='-', text=" "), end=' ')

# Print intersection-over-union scores for all classes.
def printClassScores(scoreList, instScoreList, config):
    if (config.quiet):
        return
    print(config.bold + "classes          IoU      nIoU" + config.nocol)
    print("--------------------------------")
    for label in config.evalLabels:
        if (id2label[label].ignoreInEval):
            continue
        labelName = str(id2label[label].name)
        iouStr = getColorEntry(scoreList[labelName], config) + "{val:>5.3f}".format(val=scoreList[labelName]) + config.nocol
        niouStr = getColorEntry(instScoreList[labelName], config) + "{val:>5.3f}".format(val=instScoreList[labelName]) + config.nocol
        print("{:<14}: ".format(labelName) + iouStr + "    " + niouStr)

# Print intersection-over-union scores for all categorys.
def printCategoryScores(scoreDict, instScoreDict, config):
    if (config.quiet):
        return
    print(config.bold + "categories       IoU      nIoU" + config.nocol)
    print("--------------------------------")
    for categoryName in scoreDict:
        if all( label.ignoreInEval for label in category2labels[categoryName] ):
            continue
        iouStr  = getColorEntry(scoreDict[categoryName], config) + "{val:>5.3f}".format(val=scoreDict[categoryName]) + config.nocol
        niouStr = getColorEntry(instScoreDict[categoryName], config) + "{val:>5.3f}".format(val=instScoreDict[categoryName]) + config.nocol
        print("{:<14}: ".format(categoryName) + iouStr + "    " + niouStr)

# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, config):
    if len(predictionImgList) != len(groundTruthImgList):
        printError("List of images for prediction and groundtruth are not of equal size.")
    confMatrix    = generateMatrix(config)
    instStats     = generateInstanceStats(config)
    perImageStats = {}
    nbPixels      = 0

    if not config.quiet:
        print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
    for i in range(len(predictionImgList)):
        predictionImgFileName = predictionImgList[i]
        groundTruthImgFileName = groundTruthImgList[i]
        #print "Evaluate ", predictionImgFileName, "<>", groundTruthImgFileName
        nbPixels += evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instStats, perImageStats, config)

        # sanity check
        if confMatrix.sum() != nbPixels:
            printError('Number of analyzed pixels and entries in confusion matrix disagree: confMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

        if not config.quiet:
            print("\rImages Processed: {}".format(i+1), end=' ')
            sys.stdout.flush()
    if not config.quiet:
        print("\n")

    # sanity check
    if confMatrix.sum() != nbPixels:
        printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

    # print confusion matrix
    if (not config.quiet):
        printConfMatrix(confMatrix, config)

    # Calculate IOU scores on class level from matrix
    classScoreList = {}
    for label in config.evalLabels:
        labelName = id2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, config)

    # Calculate instance IOU scores on class level from matrix
    classInstScoreList = {}
    for label in config.evalLabels:
        labelName = id2label[label].name
        classInstScoreList[labelName] = getInstanceIouScoreForLabel(label, confMatrix, instStats, config)

    # Print IOU scores
    if (not config.quiet):
        print("")
        print("")
        printClassScores(classScoreList, classInstScoreList, config)
        iouAvgStr  = getColorEntry(getScoreAverage(classScoreList, config), config) + "{avg:5.3f}".format(avg=getScoreAverage(classScoreList, config)) + config.nocol
        niouAvgStr = getColorEntry(getScoreAverage(classInstScoreList , config), config) + "{avg:5.3f}".format(avg=getScoreAverage(classInstScoreList , config)) + config.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # Calculate IOU scores on category level from matrix
    categoryScoreList = {}
    for category in category2labels.keys():
        categoryScoreList[category] = getIouScoreForCategory(category,confMatrix,config)

    # Calculate instance IOU scores on category level from matrix
    categoryInstScoreList = {}
    for category in category2labels.keys():
        categoryInstScoreList[category] = getInstanceIouScoreForCategory(category,confMatrix,instStats,config)

    # Print IOU scores
    if (not config.quiet):
        print("")
        printCategoryScores(categoryScoreList, categoryInstScoreList, config)
        iouAvgStr = getColorEntry(getScoreAverage(categoryScoreList, config), config) + "{avg:5.3f}".format(avg=getScoreAverage(categoryScoreList, config)) + config.nocol
        niouAvgStr = getColorEntry(getScoreAverage(categoryInstScoreList, config), config) + "{avg:5.3f}".format(avg=getScoreAverage(categoryInstScoreList, config)) + config.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # write result file
    allResultsDict = createResultDict( confMatrix, classScoreList, classInstScoreList, categoryScoreList, categoryInstScoreList, perImageStats, config )
    writeJSONFile( allResultsDict, config)

    # return confusion matrix
    return allResultsDict

# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instanceStats, perImageStats, config):
    # Loading all resources for evaluation.
    try:
        predictionImg = Image.open(predictionImgFileName)
        predictionNp  = np.array(predictionImg)
    except:
        printError("Unable to load " + predictionImgFileName)
    try:
        groundTruthImg = Image.open(groundTruthImgFileName)
        groundTruthNp = np.array(groundTruthImg)
    except:
        printError("Unable to load " + groundTruthImgFileName)
    # load ground truth instances, if needed
    if config.evalInstLevelScore:
        groundTruthInstanceImgFileName = groundTruthImgFileName.replace("semantic","instance")
        try:
            instanceImg = Image.open(groundTruthInstanceImgFileName)
            instanceNp  = np.array(instanceImg)
        except:
            printError("Unable to load " + groundTruthInstanceImgFileName)

    # Check for equal image sizes
    if (predictionImg.size[0] != groundTruthImg.size[0]):
        printError("Image widths of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
    if (predictionImg.size[1] != groundTruthImg.size[1]):
        printError("Image heights of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
    if ( len(predictionNp.shape) != 2 ):
        printError("Predicted image has multiple channels.")

    imgWidth  = predictionImg.size[0]
    imgHeight = predictionImg.size[1]
    nbPixels  = imgWidth*imgHeight

    # Evaluate images
    if (CSUPPORT):
        # using cython
        confMatrix = addToConfusionMatrix.cEvaluatePair(predictionNp, groundTruthNp, confMatrix, config.evalLabels)
    else:
        # the slower python way
        for (groundTruthImgPixel,predictionImgPixel) in izip(groundTruthImg.getdata(),predictionImg.getdata()):
            if (not groundTruthImgPixel in config.evalLabels):
                printError("Unknown label with id {:}".format(groundTruthImgPixel))

            confMatrix[groundTruthImgPixel][predictionImgPixel] += 1

    if config.evalInstLevelScore:
        # Generate category masks
        categoryMasks = {}
        for category in instanceStats["categories"]:
            categoryMasks[category] = np.in1d( predictionNp , instanceStats["categories"][category]["labelIds"] ).reshape(predictionNp.shape)

        instList = np.unique(instanceNp[instanceNp%256 > 0])
        for instId in instList:
            labelId = int(instId//256)
            label = id2label[ labelId ]
            if label.ignoreInEval:
                continue

            mask = instanceNp==instId
            instSize = np.count_nonzero( mask )

            tp = np.count_nonzero( predictionNp[mask] == labelId )
            fn = instSize - tp

            weight = config.avgClassSize[label.name] / float(instSize)
            tpWeighted = float(tp) * weight
            fnWeighted = float(fn) * weight

            instanceStats["classes"][label.name]["tp"]         += tp
            instanceStats["classes"][label.name]["fn"]         += fn
            instanceStats["classes"][label.name]["tpWeighted"] += tpWeighted
            instanceStats["classes"][label.name]["fnWeighted"] += fnWeighted

            category = label.category
            if category in instanceStats["categories"]:
                catTp = 0
                catTp = np.count_nonzero( np.logical_and( mask , categoryMasks[category] ) )
                catFn = instSize - catTp

                catTpWeighted = float(catTp) * weight
                catFnWeighted = float(catFn) * weight

                instanceStats["categories"][category]["tp"]         += catTp
                instanceStats["categories"][category]["fn"]         += catFn
                instanceStats["categories"][category]["tpWeighted"] += catTpWeighted
                instanceStats["categories"][category]["fnWeighted"] += catFnWeighted

    if config.evalPixelAccuracy:
        notIgnoredLabels = [l for l in config.evalLabels if not id2label[l].ignoreInEval]
        notIgnoredPixels = np.in1d( groundTruthNp , notIgnoredLabels , invert=True ).reshape(groundTruthNp.shape)
        erroneousPixels = np.logical_and( notIgnoredPixels , ( predictionNp != groundTruthNp ) )
        perImageStats[predictionImgFileName] = {}
        perImageStats[predictionImgFileName]["nbNotIgnoredPixels"] = np.count_nonzero(notIgnoredPixels)
        perImageStats[predictionImgFileName]["nbCorrectPixels"]    = np.count_nonzero(erroneousPixels)

    return nbPixels

# The main method
def main(argv):
    global config

    predictionImgList = []
    groundTruthImgList = []

    if len(argv) >= 1 :
        config.predictionPath = os.path.abspath(argv[0])
        config.exportFile = os.path.join(config.predictionPath,"resultPixelLevelSemanticLabeling.json")
        if len(argv) == 2 :
            config.groundTruthSearch  = os.path.abspath(os.path.join(argv[1],'*.png'))

        # use the ground truth search string specified above
        groundTruthImgList = glob.glob(config.groundTruthSearch)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(config.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
        for gt in groundTruthImgList:
            predictionImgList.append( getPrediction(config,gt) )

        # evaluate
        evaluateImgLists(predictionImgList, groundTruthImgList, config)
    else:
        printError("Wrong number of input arguments. Please input the path to the result folder.")
    return


# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
