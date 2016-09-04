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
print "Getting experience_count_of_features( priorfeatureE )..."
priorfeatureE = crf.getpriorfeatureE(corps, featureTS)  # calculate experiment_count_of_feature.
print "Done."

print "Initializing weights on Uniform distribution..."
weights = np.random.uniform(0, 1, K)
weights /= np.sum(weights) # normalization of weights
print "Done"

print "learning weights by Maximum Likelihood Estimation..."
weights, negloglikelifuncvalue = lbf.lbfgs(weights, corps, featureTS, words2tagids, priorfeatureE, 10, 40)
print "Done. We have got weights of features by Maximum Likelihood Estimation."

corp = "Dingyang has ate ."
print "The sentence are:"
print corp
print "the tags of this sentence with maximum probability are: "
log_Phi, log_z = crf.getlogfactor_Phi_ofacorp(weights, corp, featureTS, words2tagids)
max_value, Y_index = crf.Viterbi(log_Phi, len(corp)-2)
Y_list= []
for i in range(len(Y_index)):
    Y_list.append(Y_index[i]+" ")
print Y_list