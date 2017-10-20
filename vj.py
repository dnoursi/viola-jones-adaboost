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

# $0 io
def getSingleImage(filename):
    f = Image.open(filename)
    assert f.size == (64,64)
    # make b/w
    f = f.convert('L')
    result = np.array(f.getdata())
    result = np.reshape(result, (64, 64))
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

# The crux of operations
# Responsible for sliding a certain sized window over all possible parts of img
def slideTwoBoxesAcross(xLen, yLen, xWindow, yWindow, xStride=1, yStride=1):
    assert xLen >= xWindow >= xStride
    assert yLen >= yWindow >= yStride

    xRangeSize = xLen - xWindow
    yRangeSize = yLen - yWindow

    result = []
    for ix in range(0, xRangeSize, xStride):
        for iy in range(0, yRangeSize, yStride):
            if xWindow % 2 == 0:
                result.append(splitBoxTwo(ix, iy, xWindow, yWindow, True))
            if yWindow % 2 == 0:
                result.append(splitBoxTwo(ix, iy, xWindow, yWindow, False))
    return result

# ensures both window edge lengths are even numbers
# the 'windows' in this program are the total window:
# only specified in size, and cover the space of both rectangles in the feature
def allTwoBoxFeatures(xLen, yLen):
    result = []
    for xWindow in range(2, xLen, 2):
    #for xWindow in range(2, xLen, 1):
        for yWindow in range(2, yLen, 2):
            print("xy:", xWindow, yWindow)
        #for yWindow in range(2, yLen, 1):
            result += slideTwoBoxesAcross(xLen, yLen, xWindow, yWindow)
    return result

def computeFeature(iimages, featuretbl, imageIndex, featureIndex):
    image = iimages[imageIndex]
    feature = featuretbl[featureIndex]

    x1 = feature[0]
    y1 = feature[1]
    x2 = feature[2]
    y2 = feature[3]
    box1 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1]

    x1 = feature[4]
    y1 = feature[5]
    x2 = feature[6]
    y2 = feature[7]
    box2 = image[x2][y2] + image[x1][y1] - image[x1][y2] - image[x2][y1]

    return box2 - box1

# $3 prediction

# Ordinarypolarity is a bool
# given: a specific image, a [feature, threshold, polarity] aka weak-stump-classifier
# returns: \pm 1, a label prediction
def predictWeak(iimages, featuretbl, imageIndex, featureIndex, threshold, ordinaryPolarity):
    rawValue = computeFeature(iimages, featuretbl, imageIndex, featureIndex)
    result = 1 if rawValue > threshold else -1
    result *= 1 if ordinaryPolarity else -1
    return result

# given: a specific image, and one boosted classifier, which is a list of [weak-stump, alpha]
# returns: either every vote from each weak classifier,  or just \pm 1
def predictAgg(iimages, featuretbl, imageIndex, boostedClassifier, returnScaledVotes=False):
    scaledVotes = []
    for elem in boostedClassifier:
        classifier = elem[0]
        alpha = elem[1]
        featureIndex = classifier[0]
        threshold = classifier[1]
        polarity = classifier[2]
        prediction = predictWeak(iimages, featuretbl, imageIndex, featureIndex, threshold, polarity)
        scaledVotes.append(prediction * alpha)
    if returnScaledVotes:
        return scaledVotes

    return 1 if sum(scaledVotes) >= 0 else -1

# $4 adaboost

# Best weak learner on the training data we currently care about
def bestLearner(iimages, iindices, imageLabels, featuretbl, weights):
    iimages = [iimages[i] for i in iindices]
    ntrain = len(iimages)
    assert ntrain == len(imageLabels)
    nfeat = len(featuretbl)
    # Find each features maximum utility, then pick the best of those
    allFeatureValues = []
    # list of lists for each features best shot at a weak learner
    # contain error, value of j (separator), theta (threshold), ordinary polarity (bool)
    for jfeat in range(nfeat):
        eachImage = []
        for jim in range(ntrain):
            vl = computeFeature(iimages, featuretbl, jim, jfeat)
            eachImage.append(vl)
        allFeatureValues.append(eachImage)
    everyFeaturesBest = []
    for jfeat in range(nfeat):
        unsorted = np.array(allFeatureValues[jfeat])
        permutation = np.argsort(unsorted)

        permPosWeights = [(weights[index] if imageLabels[index] == 1 else 0) for index in permutation]
        permNegWeights = [(weights[index] if imageLabels[index] == -1 else 0) for index in permutation]
        permPosWeights = np.array(permPosWeights)
        permNegWeights = np.array(permNegWeights)

        imageErrors = []
        for jim in range(ntrain):
            leftPositive = sum(permPosWeights[:jim+1])
            leftNegative = sum(permNegWeights[:jim+1])
            rightPositive = sum(permPosWeights[jim+1:])
            rightNegative = sum(permNegWeights[jim+1:])

            if leftPositive + rightNegative > leftNegative + rightPositive:
                bestError = leftNegative + rightPositive
                ordinaryPolarity = False
            else:
                bestError = leftPositive + rightNegative
                ordinaryPolarity = True

            bestError /= ntrain
            imageErrors.append([bestError, jim, ordinaryPolarity, jfeat])

        bestErrorIndex = np.argsort([x[0] for x in imageErrors])[0]
        jthresh = imageErrors[bestErrorIndex][1]
        thresh = (unsorted[permutation[jthresh]] + unsorted[permutation[jthresh+1]]) * float(1)/2
        everyFeaturesBest.append([imageErrors[bestErrorIndex], thresh])

    bestClassifierIndex = np.argsort([x[0] for x in everyFeaturesBest])[0]
    result = everyFeaturesBest[bestClassifierIndex] # [err, splitimageindex, polarity, feature index] ,  thresh *@ 1
    result.append(result[0][2]) # polarity *@ 2
    result.append(result[0][3]) # feat index *@ 3
    result[0] =  result[0][0] # error @ 0
    return result

def computeAlpha(errorValue):
    return math.log((1 - errorValue) / errorValue) / 2

# ordinary polarity is bool
def newestWeights(weights, error, alpha, thresh, ordinaryPolarity, iimages, iindices, labels):
    ntrain = len(weights)
    iimages = [iimages[i] for i in iindices]
    assert ntrain == len(iimages) == len(labels)

    z = 2 * ( error * (1 - error)) ** .5
    result = []
    for i in range(ntrain):
        prediction = predictWeak(iimages, featuretbl, i, featureindex, thresh, ordinaryPolarity)
        correction = prediction * (1 if labels[i] == prediction else -1)
        result.append(weights[i] * math.exp(-alpha * correction) / z )
    return result

# Decide whether we're done boosting, and will move on in the cascade
# aka: optiize theta satisfying DR=100%, then return theta and frp at this theta
def aggCatchAll(iimages, labels, featuretbl, boostedClassifier):
    ntrain = len(labels)
    assert ntrain == len(predictions)

    predictions = [predictAgg(iimages, featuretbl, i, boostedClassifier, True) for i in range(ntrain)]
    # pp ~ predict positive
    # tp ~ true positive
    tpIndices = [i for i in range(ntrain) if labels[i] == 1]
    tpPredictions = [predictions[i] for i in tpIndices]
    # least confident
    lcPositive = np.argsort(tpPredictions)[0]
    theta = tpPredictions[lcPositive] - abs(tpPredictions[lcPositive]) * .04
    ppIndices = [i for i in tpIndices if predictions[i] > theta]
    nfp = sum([ 1 for i in ppIndices if labels[i] != 1])
    fpr = nfp / len(ppIndices)
    return [fpr, theta, ppIndices]

def pickleWrapper(request=None, filename="classifier.pkl"):
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


def main():
    dev = True

    trainPositive = getAllDirImages("data/faces")
    trainNegative = getAllDirImages("data/background")
    if dev:
        trainPositive = trainPositive[:50]
        trainNegative = trainNegative[:50]
        print([len(elem) for elem in trainPositive])
        print(trainPositive)
        print(trainNegative)

    # ideally this is feature1tbl
    #if not featuretbl = pickleWrapper(None, "featuretbl"):

    featuretbl = pickleWrapper(None, "featuretbl")
    if featuretbl == False:
        featuretbl = allTwoBoxFeatures(64,64)
        pickleWrapper(featuretbl, "featuretbl")

    # element of feature1tbl is 2 adjacent rectangles
    # element of feature2tbl is 3 adjacent rectangles
    # element of feature3tbl is 2 pairs of diagonal rectangles
    # adjacent rectangles are of identical size and dimension
    # store featuretypes as a third global var: values are either 0,1,2

    cascade = []
    fprSufficient = .30

    trainData = trainPositive + trainNegative
    trainLabels = [1 for _ in range(len(trainPositive))] + [-1 for _ in range(len(trainNegative))]

    iimages = pickleWrapper(None, "iimages")
    if iimages == False:
        iimages = [constructIntegralImage(elem) for elem in trainData]
        pickleWrapper(iimages, "iimages")

    iindices = list(range(len(iimages)))

    for _ in range(4):
        ntrain = len(iindices)
        scalar = float(1)/ntrain
        weights = [scalar for _ in range(ntrain)]
        fpr = fprSufficient + 1
        boostedAgg = []
        while fpr > fprSufficient:

            weak = bestLearner(iimages, iindices, trainLabels, featuretbl, weights)
            print(weak)
            error = weak[0]
            threshold = weak[1]
            ordinaryPolarity = weak[2]
            featureIndex = weak[3]
            alpha = computeAlpha(error)
            weights = updateWeights(weights, error, alpha, threshold, ordinaryPolarity, iimages, iindices, labels)
            boostedAgg.append([[featureIndex, threshold, ordinaryPolarity], alpha])

            aggInfo = aggCatchAll(iindices, trainLabels, featuretbl, boostedAgg)
            fpr = aggInfo[0]
            if fpr > fprSufficient:
                theta = aggInfo[1]
                cascade.append([boostedAgg, theta])
                iindices = aggInfo[2]
                # break

    pickleWrapper(cascade)

if __name__ == "__main__":
    main()
