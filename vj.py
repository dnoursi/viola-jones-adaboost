#!/usr/bin/env python3

# Viola Jones
# David Noursi Stat 277

""" potential extensions

variance-normalizing each image
reflecting each face to double the amount of training data
just more features (extra credit)

"""

import numpy as np
from PIL import Image
import os
import pickle as pkl
import math

# $0 io
def getSingleImage(filename, xDim=64, yDim=64):
    f = Image.open(filename)

    if xDim and yDim:
        assert f.size == (xDim, yDim)
    else:
        sz = f.size
        xDim = sz[0]
        yDim = sz[1]
    # make b/w
    f = f.convert('L')
    result = np.array(f.getdata())
    result = np.reshape(result, (xDim, yDim))
    return result
    # alt methods:
    # pixelMatrx = f.load()
    # grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

def getAllDirImages(dirname):
    return [getSingleImage(os.path.join(dirname,filename)) for filename in os.listdir(dirname) if (filename.endswith("jpg") and not filename.startswith("."))] # if os.path.isfile(filename)

# $1 iimages

# Do all sums to get ii for a single image
# data is the pixel values for an image
def constructIntegralImage(data):
    print("iimage")
    result = [data[0]]
    # Consider data to be a list of rows,
    # Rows descend down the image
    for row in data[1:]:
        # newii = [np.cumsum(row[:i+1]) for i in range(len(row))]
        # step1 Sum over this row
        newii = []
        rt = 0 # running total
        for pixel in row:
            rt += pixel
            newii.append(rt)
        # step2 Now compute actual ii
        previi = result[-1]
        assert len(previi) == len(newii)
        result.append([ newii[index] + previi[index] for index in range(len( previi)) ])

    return result

# $2 featuretbl

"""
x is across rows
aka x is the first and outer index
images are accesed as pixels[x][y]

this means x is down
and y is to the right
"""

# the splitting of a window into two, either possible way
# splitX is bool
def splitBoxTwo(ix, iy, xWindow, yWindow, splitOnX):
    xEnd = ix + xWindow
    yEnd = iy + yWindow
    feature = []

    if splitOnX:
        assert xWindow % 2 == 0
        # Separated horizontally (perpendicular to vertical x axis)
        xMid = ix + int(xWindow/2)
        feature += [ix, iy]
        feature += [xMid, yEnd]
        feature += [xMid, iy]
        feature += [xEnd, yEnd]
    else:
        assert yWindow % 2 == 0
        yMid = iy + int(yWindow/2)
        feature += [ix, iy]
        feature += [xEnd, yMid]
        feature += [ix, yMid]
        feature += [xEnd, yEnd]

    return feature


# there exist 1,2,3,4, 1',2',3',4'
# store 1,2,3 and 2',3',4'
# ergo boxes are from feature elems [1 and 4] together with [3 and 6]
# minus mid box 2 and 5
def splitBoxThree(ix, iy, xWindow, yWindow, splitOnX):
    xEnd = ix + xWindow
    yEnd = iy + yWindow
    feature = []

    if splitOnX:
        assert xWindow % 3 == 0
        # Separated horizontally (perpendicular to vertical x axis)
        xThird = int(xWindow/3)
        xMid1 = ix + xThird
        #xMid2 = ix + 2 * int(xWindow/3)
        xMid2 = xMid1 + xThird

        feature += [ix, iy]
        feature += [xMid1, yEnd]

        feature += [xMid1, iy]
        feature += [xMid2, yEnd]

        feature += [xMid2, iy]
        feature += [xEnd, yEnd]

    else:
        assert yWindow % 3 == 0
        yThird = int(yWindow/3)
        yMid1 = iy + yThird
        #xMid2 = ix + 2 * int(xWindow/3)
        yMid2 = yMid1 + yThird

        feature += [ix, iy]
        feature += [xEnd, yMid1]

        feature += [ix, yMid1]
        feature += [xEnd, yMid2]

        feature += [ix, yMid2]
        feature += [xEnd, yEnd]

    return feature

# topleft, topmid, topright, midleft, midmid, midright, botleft, botmid, botright
def splitBoxFour(ix, iy, xWindow, yWindow):
    xEnd = ix + xWindow
    yEnd = iy + yWindow
    feature = []

    assert yWindow % 2 == 0
    yMid = iy + int(yWindow/2)

    feature += [ix, iy]
    feature += [ix, yMid]
    feature += [ix, yEnd]

    assert xWindow % 2 == 0
    # Separated horizontally (perpendicular to vertical x axis)
    xMid = ix + int(xWindow/2)

    feature += [xMid, iy]
    feature += [xMid, yMid]
    feature += [xMid, yEnd]

    feature += [xEnd, iy]
    feature += [xEnd, yMid]
    feature += [xEnd, yEnd]

    return feature



def splitBoxN(n, ix, iy, xWindow, yWindow, splitOnX):
    if n == 4:
        assert splitOnX == None
        return splitBoxFour(ix, iy, xWindow, yWindow)
    if n == 2:
        return splitBoxTwo(ix, iy, xWindow, yWindow, splitOnX)
    if n == 3:
        return splitBoxThree(ix, iy, xWindow, yWindow, splitOnX)

# The crux of operations
# Responsible for sliding a certain sized window over all possible parts of img
def slideNBoxesAcross(n, splitOnX, xLen, yLen, xWindow, yWindow, xStride=1, yStride=1):
    assert xLen >= xWindow >= xStride
    assert yLen >= yWindow >= yStride
    xRangeSize = xLen - xWindow
    yRangeSize = yLen - yWindow

    result = []
    for ix in range(0, xRangeSize, xStride):
        for iy in range(0, yRangeSize, yStride):
            result.append(splitBoxN(n, ix, iy, xWindow, yWindow, splitOnX))

    tuples = []
    for elem in result:
            tuples.append([elem, n])

    return tuples


# ensures both window edge lengths are even numbers
# the 'windows' in this program are the total window:
# only specified in size, and cover the space of both rectangles in the feature
def allBoxFeatures(xLen, yLen):
    result = []
    sizes = list(range(4, 25, 2)) + list(range(28, 64, 4))
    for xWindow in sizes:
    #for xWindow in range(2, xLen, 2):
        for yWindow in sizes:
        #for yWindow in range(2, yLen, 1):
            #print("xy:", xWindow, yWindow)

            n = 2
            if xWindow % n == 0:
                result += slideNBoxesAcross(n, True, xLen, yLen, xWindow, yWindow, n, n)
            if yWindow % n == 0:
                result += slideNBoxesAcross(n, False, xLen, yLen, xWindow, yWindow, n, n)
            n = 3
            if xWindow % n == 0:
                result += slideNBoxesAcross(n, True, xLen, yLen, xWindow, yWindow, n, n)
            if yWindow % n == 0:
                result += slideNBoxesAcross(n, False, xLen, yLen, xWindow, yWindow, n, n)
            n = 4
            if xWindow % 2 == 0 and yWindow % 2 == 0:
                result += slideNBoxesAcross(n, None, xLen, yLen, xWindow, yWindow, n, n)

    return result

#def computeBox(image, x1, x2, y1, y2):
#    return image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1]

def computeFeature(iimages, imageIndex, featuretbl, featureIndex):
    image = iimages[imageIndex]
    full = featuretbl[featureIndex]
    feature = full[0]
    nfeature = full[1]

    if nfeature == 2:
        return computeTwoBoxes(image, feature)
    if nfeature == 3:
        return computeThreeBoxes(image, feature)
    if nfeature == 4:
        return computeFourBoxes(image, feature)

# points stored as
# 012
# 345
# 678
# multiply all of those by 2, because x and y
def computeFourBoxes(image, feature):
    x1 = feature[0]
    y1 = feature[1]
    x2 = feature[8]
    y2 = feature[9]
    box1 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    x1 = x2
    y1 = y2
    x2 = feature[16]
    y2 = feature[17]
    box4 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    # top right
    x1 = feature[2]
    y1 = feature[3]
    x2 = feature[10]
    y2 = feature[11]
    box2 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    # bottom left
    x1 = feature[6]
    y1 = feature[7]
    x2 = feature[14]
    y2 = feature[15]
    box3 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    return box1 + box4 - box3 - box2

def computeThreeBoxes(image, feature):
    x1 = feature[0]
    y1 = feature[1]
    x2 = feature[2]
    y2 = feature[3]
    box1 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    x1 = feature[4]
    y1 = feature[5]
    x2 = feature[6]
    y2 = feature[7]
    box2 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    x1 = feature[8]
    y1 = feature[9]
    x2 = feature[10]
    y2 = feature[11]
    box3 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    return box1 + box3 - box2

def computeTwoBoxes(image, feature):
    x1 = feature[0]
    y1 = feature[1]
    x2 = feature[2]
    y2 = feature[3]
    box1 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    x1 = feature[4]
    y1 = feature[5]
    x2 = feature[6]
    y2 = feature[7]
    box2 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1] # computeBox(image, x1, x2, y1, y2)

    return box2 - box1

# general args: [iimages, labels, iindices], [featuretbl, featureindex], thresh, polarity

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$3 prediction

# Stump predicts on a single image in iimages
# Ordinarypolarity is a bool
# given: a specific image, a [feature, threshold, polarity] aka weak-stump-classifier
# returns: \pm 1, a label prediction
def predictWeak(iimages, imageIndex, featuretbl, classifier):
    featureIndex = classifier[0]
    threshold = classifier[1]
    polarity = classifier[2]

    rawValue = computeFeature(iimages, imageIndex, featuretbl, featureIndex)
    result = (1 if rawValue >= threshold else -1)
    result *= (1 if polarity else -1)
    return result

# given: a specific image, and one boosted classifier, which is a list of [weak-stump, alpha]
# returns: sum of scaled votes from each weak classifier
def predictAgg(iimages, imageIndex, featuretbl, boostedClassifier):
    scaledVotes = []

    for elem in boostedClassifier:
        classifier = elem[0]
        alpha = elem[1]
        featureIndex = classifier[0]
        threshold = classifier[1]
        polarity = classifier[2]
        prediction = predictWeak(iimages, imageIndex, featuretbl, classifier)
        scaledVotes.append(prediction * alpha)
    ssv = sum(scaledVotes)
    #result = [ ssv, (1 if ssv >= 0 else -1)]
    #return result
    return ssv

# $4 adaboost

# Best weak learner on the training data we currently care about
def bestLearner(iimages, labels, iindices, featuretbl, weights):
    ntrain = len(iindices)
    assert ntrain == len(weights)
    flntrain = float(ntrain)
    nfeat = len(featuretbl)

    # Find each features maximum utility, then pick the best of those
    allFeatureValues = []
    # list of lists for each features best shot at a weak learner
    # contain error, value of j (separator), theta (threshold), ordinary polarity (bool)

    # Compute everything
    for jfeat in range(nfeat):
        if jfeat % 20000 == 0:
            print("compute for jfeat", jfeat)

        eachImage = []
        for jim in iindices:
            vl = computeFeature(iimages, jim, featuretbl, jfeat)
            eachImage.append(vl)
        allFeatureValues.append(eachImage)
    everyFeaturesBest = []

    # For each feature: 1 sort images by performance, then 2 select threshold/polarity
    for jfeat in range(nfeat):
        if jfeat % 20000 == 0:
            print("theta for jfeat", jfeat)
        unsorted = np.array(allFeatureValues[jfeat])
        permutation = np.argsort(unsorted)

        permPosWeights = [(weights[index] if labels[index] == 1 else 0) for index in permutation]
        #pprime = [(weights[permutation[index]] if labels[permutation[index]] == 1 else 0) for index in range(len(permutation))]
        #assert permPosWeights == pprime
        permNegWeights = [(weights[index] if labels[index] == -1 else 0) for index in permutation]
        permPosWeights = np.array(permPosWeights)
        permNegWeights = np.array(permNegWeights)

        imageErrors = []

        cachePositive = sum(permPosWeights)
        cacheNegative = sum(permNegWeights)
        leftPositive = 0
        leftNegative = 0
        for jim in range(ntrain):
            leftPositive += permPosWeights[jim]
            #leftPositive = sum(permPosWeights[:jim+1])
            leftNegative += permNegWeights[jim]
            #leftNegative = sum(permNegWeights[:jim+1])
            rightPositive = cachePositive - leftPositive
            #rightPositive = sum(permPosWeights[jim+1:])
            rightNegative = cacheNegative - leftNegative
            #rightNegative = sum(permNegWeights[jim+1:])

            if leftPositive + rightNegative > leftNegative + rightPositive:
                bestError = leftNegative + rightPositive
                ordinaryPolarity = False
            else:
                bestError = leftPositive + rightNegative
                ordinaryPolarity = True

            #bestError /= flntrain
            imageErrors.append([bestError, jim, ordinaryPolarity, jfeat])

        bestErrorSort = np.argsort([x[0] for x in imageErrors])
        bestErrorIndex = bestErrorSort[0]
        #print(bestErrorIndex)
        jthresh = imageErrors[bestErrorIndex][1]
        thresh = (unsorted[permutation[jthresh]])#+ unsorted[permutation[jthresh+1]]) * float(1)/2
        everyFeaturesBest.append([imageErrors[bestErrorIndex], thresh])

    print(everyFeaturesBest)
    bestClassifierSort = np.argsort([x[0][0] for x in everyFeaturesBest])
    bestClassifierIndex = bestClassifierSort[0]
    print(bestClassifierIndex)

    result = everyFeaturesBest[bestClassifierIndex] # [err, splitimageindex, polarity, feature index] ,  thresh *@ 1
    result.append(result[0][2]) # polarity *@ 2
    result.append(result[0][3]) # feat index *@ 3
    result[0] =  result[0][0] # error @ 0
    return result

def computeAlpha(errorValue):
    return math.log((1 - errorValue) / errorValue) / 2

def assertWeightMargin(sweights, margin):
    assert abs(sweights - 1) < margin

#def updateWeights(weights, error, alpha, thresh, ordinaryPolarity, iimages, iindices, labels, featuretbl, featureIndex):
# ordinary polarity is bool
def updateWeights(iimages, labels, iindices, featuretbl, weights, weak, alpha):
    ntrain = len(weights)
    assert ntrain == len(iindices)
    print(sum(weights))
    margin = .8
    assertWeightMargin(sum(weights), margin)
    #assert ntrain == len(iindices)
    error = weak[0]
    threshold = weak[1]
    ordinaryPolarity = weak[2]
    featureIndex = weak[3]

    z = 2 * (( error * (1 - error)) ** .5)
    result = []
    for i in range(ntrain):
        prediction = predictWeak(iimages, iindices[i], featuretbl, [featureIndex, threshold, ordinaryPolarity])
        correction = (1 if labels[iindices[i]] != prediction else -1)
        result.append(weights[i] * math.exp(alpha * correction) / z )
    print(sum(result))
    assertWeightMargin(sum(weights), margin)
    return result

# Decide whether we're done boosting, and will move on in the cascade
# aka: optiize theta satisfying DR=100%, then return theta and frp at this theta
def aggCatchAll(iimages, labels, iindices, featuretbl, boostedClassifier):
    ntrain = len(iindices)
    predictions = [predictAgg(iimages, i, featuretbl, boostedClassifier) for i in iindices]
    print("Agg catch all: predictions", predictions)
    assert ntrain == len(predictions)

    # pp aka predicted positive
    # tp aka true positive
    tpIndices = [i for i in iindices if labels[i] == 1]
    assert len(tpIndices) == len(iimages)/2
    tpPredictions = [predictions[i] for i in range(len(tpIndices))]
    # least confident
    confidenceSort = np.argsort(tpPredictions)
    print(confidenceSort, tpIndices, "oof")
    lcPositive = tpIndices[confidenceSort[0]]
    theta = tpPredictions[lcPositive] - (abs(tpPredictions[lcPositive]) * .04)

    ppIndices = [ iindices[i] for i in range(ntrain) if predictions[i] > theta ]
    ntn = len(iindices) - len(tpIndices)
    nfp = len([ i for i in ppIndices if labels[i] != 1])
    # len almost equiv to sum
    fpr = nfp / ntn
    return [fpr, theta, ppIndices]

def pickleWrapper(filename, request=None):
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    if request:
        with open(filename, "wb") as output:
            pkl.dump(request, output)
    else:
        try:
            with open(filename, "rb") as output:
                result = pkl.load(output)
                return result
        except IOError:
            return False

def findFaces(iimage, featuretbl, cascade, xStride=128, yStride=128):
    xWindow = len(iimage)
    yWindow = len(iimage[0])
    assert xWindow >= xStride
    assert yWindow >= yStride

    result = []
    for ix in range(0, xWindow, xStride):
        for iy in range(0, yWindow, yStride):
            faceCoords = topFace(iimage, featuretbl, cascade, ix, iy, xStride, yStride)
            if faceCoords:
                result.append(faceCoords)

    return result

import time
# find the absolutebest face within xWin x yWin
# prevents exessive overlap: within this function, there is strictly none
def topFace(iimage, featuretbl, cascade, ix, iy, xWindow, yWindow):
    unsorted = []
    frameSize = 64
    for jx in range(ix, ix + xWindow - frameSize, 4):
        for jy in range(iy, iy + yWindow - frameSize, 4):
            val = cascadeValue(iimage, featuretbl, cascade, jx, jy, frameSize, frameSize)
            unsorted.append([val, [jx, jy]])


    # VERY weird out of order for these three prints ...
    time.sleep(6)
    unsortedValues = [xx[0] for xx in unsorted]
    time.sleep(6)
    permutation = np.argsort(unsortedValues)
    time.sleep(6)
    print("permute should be", permutation, "at topFace invoked with ixiy", ix, iy)
    print("unsorted should be", unsorted)
    assert len(permutation) > 0
    maximum = permutation[-1]
    print("max index", maximum)
    coords = unsorted[maximum][1]
    maxValue = unsorted[maximum][0]
    print("the coords of face are", coords, "with value", maxValue)
    # SLEEPING to fix


    if maxValue > 0: # actually classified as  a face
        return coords
    return False

def cascadeValue(iimage, featuretbl, cascade, ix, iy, xFrameSize, yFrameSize):
    assert len(iimage) >= ix + xFrameSize
    assert len(iimage[0]) >= iy + yFrameSize
    cumulative = 0
    for boosted in cascade:
        boostedClassifer = boosted[0]
        theta = boosted[1]
        mini = np.array(iimage)
        mini = mini[ix: ix + xFrameSize, iy: iy + yFrameSize]
        pa = predictAgg([mini], 0, featuretbl, boostedClassifer)
        if pa < theta:
            print("$ pa", pa, "$ theta", theta, "$ boosted", boosted, "cumul", cumulative)
            return -1 # pa?
        cumulative += pa
    return cumulative


def main():
    runTestCase()

def runTestCase():

    pickleName = "featuretbl"
    featuretbl = pickleWrapper(pickleName)
    if featuretbl == False:
        featuretbl = getFeatureTable()
        pickleWrapper(pickleName, featuretbl)

    pickleName = "cascade"
    cascade = pickleWrapper(pickleName)
    if cascade == False:
        cascade = trainCascade(featuretbl)
        pickleWrapper(pickleName, cascade)

    print(cascade)
    testCase = getSingleImage("data/class.jpg", None, None)
    iitc = constructIntegralImage(testCase)
    print(len(iitc), len(iitc[0]))

    testing = topFace(iitc, featuretbl, cascade, 400, 330, 200, 200)
    print(testing)
    return

    detections = findFaces(iitc, featuretbl, cascade)
    print(len(detections), detections)

def getFeatureTable():
    return allBoxFeatures(64,64)

def trainCascade(featuretbl):
    dev = True
    devSize = 1200

    # featuretbl
    if dev:
        featuretbl = [featuretbl[i] for i in range(len(featuretbl)) if i % 7 == 0]

    # iimages
    trainLabels = [1 for _ in range((devSize if dev else 2000))] + [-1 for _ in range((devSize if dev else 2000))]

    pickleName = "iimages"
    iimages = pickleWrapper(pickleName)
    if iimages == False:
        trainPositive = getAllDirImages("data/faces")
        trainNegative = getAllDirImages("data/background")
        if dev:
            trainPositive = trainPositive[:devSize]
            trainNegative = trainNegative[:devSize]
        trainData = trainPositive + trainNegative
        trainLabels = [1 for _ in range(len(trainPositive))] + [-1 for _ in range(len(trainNegative))]

        iimages = [constructIntegralImage(elem) for elem in trainData]
        pickleWrapper(pickleName, iimages)

    iindices = list(range(len(iimages)))

    # ideally this is feature1tbl
    #if not featuretbl = pickleWrapper(None, "featuretbl"):
    # adjacent rectangles are of identical size and dimension
    cascade = []
    fprSufficient = .30
    niterations = 0
    pickleName = "cascade0"
    for createCascadeIndex in range(3):
        print("On cascade", createCascadeIndex)
        print("Cascade is", cascade)

        ntrain = len(iindices)
        scalar = float(1)/ntrain
        weights = [scalar for _ in range(ntrain)]
        boostedAgg = []

        fpr = fprSufficient + 1
        while fpr > fprSufficient:
            niterations += 1
            weak = bestLearner(iimages, trainLabels, iindices, featuretbl, weights)
            print("Weak is", weak, "iteration number", niterations, "len iindices", len(iindices))
            error = weak[0]
            threshold = weak[1]
            ordinaryPolarity = weak[2]
            featureIndex = weak[3]

            alpha = computeAlpha(error)
            boostedAgg.append([[featureIndex, threshold, ordinaryPolarity], alpha])

            weights = updateWeights(iimages, trainLabels, iindices, featuretbl, weights, weak, alpha)


            print("Bossted agg is", boostedAgg)
            aggInfo = aggCatchAll(iimages, trainLabels, iindices, featuretbl, boostedAgg)
            print("Agg info is", aggInfo)
            fpr = aggInfo[0]
            if fpr < fprSufficient:
                theta = aggInfo[1]
                cascade.append([boostedAgg, theta])

                iindices = aggInfo[2]

                pickleName = pickleName[:-1]
                pickleName += str(createCascadeIndex)
                pickleWrapper(pickleName, cascade)

    return cascade

if __name__ == "__main__":
    main()
