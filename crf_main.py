# coding=utf-8
import crf
import numpy as np
import L_BFGS as lbf

print "Reading corps..."
corps, tagids = crf.read_corps() # read train_data and organization.
print "Done."

print "Getting featureslist..."
featureTS, words2tagids = crf.getfeatureTS(corps)
print "Done."

K = np.shape(featureTS)[0]  # number of features
N = np.shape(corps)[0]  # number of train samples
print "Getting experiment_count_of_features( priorfeatureE )..."
priorfeatureE = crf.getpriorfeatureE(corps, featureTS)  # calculate experiment_count_of_feature.
print "Done."

print "Initializing uniform weights ..."
weights = np.random.uniform(0, 1, K)
weights /= np.sum(weights) # normalization of weights
print "Done"
print "Running l-bfgs..."


weights, likelyfuncvalue = lbf.lbfgs(weights, corps, featureTS, words2tagids, priorfeatureE, 10, 40)
