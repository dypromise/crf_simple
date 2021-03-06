# coding=utf-8
from collections import defaultdict
import numpy as np
import Logspace

logspace = Logspace.Logspace()


def sentence2corp(sentence):
    sentence = "<S> " + sentence.strip() + " <S>"
    return sentence.strip().split(' ')


def read_corps(corpsfile='test.txt'):
    """

    :param corpsfile: train dataset in format described following website.
    http://www.chokkan.org/software/crfsuite/tutorial.html
    :return: corps and tagids.
    """

    tagids = defaultdict(lambda: len(tagids))
    tagids["<S>"] = 0

    corps = []
    onesentence = []
    words = ["<S>"]
    tags = [0]
    with open(corpsfile, 'r') as f:
        for line in f:  # every line is one word like "Institute NNP I-NP"
            if len(line) <= 1:
                pass
            elif line != '. . O\n':  # token '. . O\n' means the end of a sentence，
                # if the word is not the start or end token <'s'> , it will be added into the list:onesentence.
                onesentence.append(line)
            else:  # If one sentence is in the end, all words in this sentence will be proceessed,
                # and store the process result into list : corps.
                for texts in onesentence:  # for each word in list 'onesentence'
                    w_t = texts.strip().split(" ")  # strip the blankspace between words, and make it a array.
                    try:
                        # because the style of number in a sentence is diverse, e.g. 100 and 1,000,000
                        # Here will detect it and replace it with '#CD#'.
                        float(w_t[0].strip().replace(',', ''));
                        words.append('#CD#')
                    except:
                        words.append(w_t[0].lower())
                    # if w_t[1] in{ '``',',',"''",'$','#',')','('}:
                    #    print w_t
                    tags.append(tagids[w_t[1]])  # The list tags contains the index of this tag in tagids.
                words.append("<S>")  # add the end token.
                tags.append(0)  # add the tag index of end token.
                if np.shape(words)[0] > 2:  # if not empty sentence
                    corps.append((words, tags))  # corps is a tuple!

                # Reinitialize the lists in order to process the next sentence.
                onesentence = []
                words = ["<S>"]
                tags = [0]
    return corps, tagids


def getfeatureTS(corps):
    featuresets = set()  # set of features.
    featureT = []  # Transfer feature list, e.g. the tuple ('T', 2, 3) means the feature:
    # word tagid tranfer from 2 to 3.or in other words it means the local tagid feature : tagid 2 is following with 3.
    featureS = []  # state feature list, feature ('S','Confidence', 1) means the word of local index i in one sentence
    # is 'Confidence' and its tagid is 1.it describe the contact of one word and its tag.
    for corp in corps:  # corp is a turple : (wordsListofaSentence, tagslist)
        for i in range(np.shape(corp[0])[0]):
            if corp[0][i] == '<S>':
                continue
            if ('S', corp[0][i], corp[1][i]) not in featuresets:
                featuresets.add(('S', corp[0][i], corp[1][i]))
                featureS.append(('S', corp[0][i], corp[1][i]))
            if corp[0][i - 1] != '<S>':
                if ('T', corp[1][i - 1], corp[1][i]) not in featuresets:
                    featuresets.add(('T', corp[1][i - 1], corp[1][i]))
                    featureT.append(('T', corp[1][i - 1], corp[1][i]))
    featureTS = featureT + featureS
    words2tagids = words2tagidfromfeatureS(featureS)
    return featureTS, words2tagids


def words2tagidfromfeatureS(featureS):
    """

    Statistics taglist of every word.
    :param featureS:
    :return: it return a dict : { "word1":[state1,state2,..] , ..}
    """

    words2tagids = {}
    for feature in featureS:
        word = feature[1]
        state = feature[2]
        if word in words2tagids:
            words2tagids[word].append(state)
        else:
            words2tagids[word] = [state]  # dict : { "word1":[state1,state2,..] , ..}

    return words2tagids


def getlogfactor_Phi_ofacorp(weights, corps_i, featureTS, words2tagids):
    """
    Given Theta_k(weights of features), and corp, we can calculate the P(y|x=corp) (y is tag list of x),
    this is the joint distribution of y conditioned on x = corp. and this distribution is the multiple multiplication of
     factor 'phi'.

    This method calculate the all factors 'Phi' of a corp.
    Factor Phi means the "Pseudo Probability" of local target variables.
    :param weights: the weights of features located in power of exp.please reference the formula of factor 'phi'.
    :param corps_i: the i_th corp.
    :param featureTS:
    :param words2tagids:the dict : { "word1":[state1,state2,..] , ..}
    :return:
    """
    corp = corps_i[0][1:-1]  # words of a sentence except start and end token<S>.
    lencorp = np.size(corp)  # the number of words in corp
    log_Phi = {}  # { 'phi':['','',...] , 'states':[[..],[],..] , 'states_num':[,...]}
    log_Phi['phi'] = [''] * (lencorp)
    log_Phi['states'] = [words2tagids[corp[i]] for i in range(lencorp)]  # [[state1,state2...],[state1,...],...]
    log_Phi['states_num'] = [np.size(words2tagids[corp[i]]) for i in range(lencorp)]  # [ stateofword1_num,...]
    for i in range(lencorp):
        if i == 0:
            d = log_Phi['states_num'][0]  # Number of labels of word corp[0]
            log_Phi['phi'][i] = np.zeros((1, d))  # the first matrix Phi['phi'] just contains state feature.
            for j in range(d):
                log_Phi['phi'][i][0, j] = weights[featureTS.index(('S', corp[i], log_Phi['states'][i][j]))]
            continue

        # if not the first factor Phi, it forms in a matrix.
        log_Phi['phi'][i] = np.zeros((log_Phi['states_num'][i - 1], log_Phi['states_num'][i]))
        for d1 in range(log_Phi['states_num'][i - 1]):
            for d2 in range(log_Phi['states_num'][i]):
                state_i_pre = log_Phi['states'][i - 1][d1]
                state_i = log_Phi['states'][i][d2]
                try:
                    Sweight = weights[featureTS.index(('S', corp[i], state_i))]
                except:
                    Sweight = 0.0
                try:
                    Tweight = weights[featureTS.index(('T', state_i_pre, state_i))]
                except:
                    Tweight = 0.0
                log_Phi['phi'][i][d1, d2] = Sweight + Tweight
    # calculate normalization coefficient of joint prob distribution of one corp.
    log_z = np.array([[0.0]])
    for i in range(lencorp):
        log_z = logspace.logspacematprod(log_z, log_Phi['phi'][i])
    log_z = logspace.logspacematprod(log_z, np.zeros((np.shape(log_z)[1], 1)))
    return log_Phi, log_z


def ForwardBackwardAlg_getlogAlphaBetalist(log_Phi, lencorp):
    """
    since each corp has a joint distribution, it has a Alpha,Beta list also.
    calculate Alpha,Beta list in order to calculate  Marginal probability  of local tardet variables.
    in fact, this algorithm is called "Forward-backward algorithm"
    :param log_Phi: factor 'phi' in log space.
    :param lencorp: the number of words in one corp
    :return:
    """
    log_Alpha = [''] * (lencorp + 1)
    log_Beta = [''] * (lencorp + 1)
    log_Alpha[0] = np.zeros((1, 1))
    log_Beta[-1] = np.zeros((log_Phi['states_num'][-1], 1))
    # log_Alpha forms in line vector, while log_Beta forms in column vector
    for i in range(lencorp):  # [1,2,...,lencorp]
        log_Alpha[i + 1] = logspace.logspacematprod(log_Alpha[i], log_Phi['phi'][i])
    for i in range(lencorp, 0, -1):  # [lencorp,lencorp-1,...,0]
        log_Beta[i - 1] = logspace.logspacematprod(log_Phi['phi'][i - 1], log_Beta[i])
    return log_Alpha, log_Beta


def getpriorfeatureE(corps, featureTS):
    """
    When we learn the parameters of conditional prob distribution P(Y|X) (in fact they are weights of features),
    we need to minimum the negative log_likelihood function of entire training data set.
    So we need the gradients of likelihood function, it is this gradient that need calculate prior expect of features.
    calculate the experiment_count_of_feature in dataset(many sentences)
    :param corps:
    :param featureTS:
    :return: the experiment_count_of_feature in dataset
    """
    N = np.shape(corps)[0]  # sample number
    K = np.shape(featureTS)[0]  # feature number
    priorfeatureE = np.zeros(K)

    for corp in corps:
        for i in range(np.shape(corp[0])[0]):  # corp[0] is the wordlist of a sentense.
            if corp[0][i] == '<S>':
                continue
            try:
                idex = featureTS.index(('S', corp[0][i], corp[1][i]))
                priorfeatureE[idex] += 1.0
            except:
                pass
            try:
                idex = featureTS.index(('T', corp[1][i - 1], corp[1][i]))
                priorfeatureE[idex] += 1.0
            except:
                pass
    priorfeatureE /= N
    return priorfeatureE


def getpostfeatureE(weights, corps, featureTS, words2tagids):
    """
    The gradients of negative log likelihood function need the return value of this method.
    this method calculate expect counts of feature.
    in order to calculate gradient of negative log likelihood.
    :param weights:
    :param corps:
    :param featureTS:
    :param words2tagids:
    :return:
    """
    K = np.shape(featureTS)[0]
    postfeatureE = np.zeros(K)
    N = np.shape(corps)[0]

    for corpidx in range(N):
        corp = corps[corpidx][0][1:-1]  # words of a sentence except start and end token<S>.
        lencorp = np.size(corp)
        log_Phi, log_z = getlogfactor_Phi_ofacorp(weights, corps[corpidx], featureTS, words2tagids)
        log_Alpha, log_Beta = ForwardBackwardAlg_getlogAlphaBetalist(log_Phi, lencorp)

        for i in range(lencorp):
            state_num_i_pre, state_num_i = np.shape(log_Phi['phi'][i])
            for di_pre in range(state_num_i_pre):
                for di in range(state_num_i):
                    plocal = np.exp(logspace.logspacematprod(logspace.logspacematprod(
                        logspace.logspacematprod(log_Alpha[i][0][di_pre], log_Phi['phi'][i][di_pre][di])
                        , log_Beta[i + 1][di][0]), - log_z))
                    if i == 0:
                        try:
                            Sidex = featureTS.index(('S', corp[i], log_Phi['states'][i][di]))
                            postfeatureE[Sidex] += plocal
                        except:
                            pass
                    else:
                        try:
                            Sidex = featureTS.index(('S', corp[i], log_Phi['states'][i][di]))
                            postfeatureE[Sidex] += plocal
                        except:
                            pass
                        try:
                            Tidex = featureTS.index(('T', log_Phi['states'][i - 1][di_pre], log_Phi['states'][i][di]))
                            postfeatureE[Tidex] += plocal
                        except:
                            pass

    postfeatureE /= N
    return postfeatureE


def getnegloglikelihoodvalue(weights, corps, featureTS, words2tagids):
    """
    Assume we have the parameters of P(Y|X):weights, we can calculate the negative log likelihood function.
     Dring this process, each corp(sentence) has one probability P(the_tag_list_of_this_corp|this_corp), and
      the multiple multiplication of this forms likelihood.
    :param weights:
    :param corps:
    :param featureTS:
    :param words2tagids:
    :return:
    """
    K = np.shape(featureTS)[0]
    N = np.shape(corps)[0]

    negloglikilivalue = 0.0

    for corpidx in range(N):

        log_Phi, log_z = getlogfactor_Phi_ofacorp(weights, corps[corpidx], featureTS, words2tagids)
        corp = corps[corpidx][0][1:-1]
        tag = corps[corpidx][1][1:-1]
        lencorp = np.size(corp)

        P_corp_tilde = [[0.0]]

        for i in range(lencorp):
            if i == 0:
                P_corp_tilde = logspace.logspacematprod(P_corp_tilde,
                                                        log_Phi['phi'][i][0, log_Phi['states'][i].index(tag[i])])
            else:
                P_corp_tilde = logspace.logspacematprod(P_corp_tilde, log_Phi['phi'][i][
                    log_Phi['states'][i - 1].index(tag[i - 1]), log_Phi['states'][i].index(tag[i])])

        negloglikilivalue += (log_z - P_corp_tilde) / N
    return negloglikilivalue


def getgradientsofnegloglikelifunc(priorfeatureE, weights, corps, featureTS, words2tagids):
    """
    calculate the gradient of negative log likelihood function.
    :param priorfeatureE:
    :param weights:
    :param corps:
    :param featureTS:
    :param words2tagids:
    :return:
    """
    postfeatureE = getpostfeatureE(weights, corps, featureTS, words2tagids)
    return postfeatureE - priorfeatureE


def Viterbi(log_Phi, lencorp):
    """

    :param weights:
    :param corp:
    :param featureTS:
    :param words2tagids:
    :return:
    """
    maxIndexofPreX = [''] * (lencorp)  # max_index[index_of_x_t] =
    log_Beta = np.zeros((np.shape(log_Phi['phi'][-1])[1], 1))
    Y_index = [''] * lencorp
    # log_Alpha forms in line vector, while log_Beta forms in column vector
    for i in range(lencorp - 1, -1, -1):  # [1,2,...,lencorp]
        log_Beta, maxIndexofPreX[i] = logspace.logsapceProduct_Max(log_Phi['phi'][i], log_Beta)
    max_value = log_Beta
    Y_index[0] = maxIndexofPreX[0]
    for i in range(1, lencorp):
        Y_index[i] = maxIndexofPreX[i][int(Y_index[i - 1])]
    return max_value, Y_index
