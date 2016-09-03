# crf_simple
  this python files are a simple implementation of line_chain crf to POS tagging.
  this project contains four python files currently. They are:
##1) logspace.py
  this class contains two method to calculate matrix product in log space.
##2) crf.py
  this file implements several methods to calculate joint probability distribution, matrginal probability distribution, inference, learning(
  max likelihood).
##3) crf_main.py
  it runs as a overall framework to train crf.
##4) L_BFGS.py
  when to learning parameters, we need to minimum the negative log_likelihood function of dataset, because it has large amount of parameters, so we need
  the limited-memery BFGS algorithm to optimize the likelihood function. 
  this file  implements the l-bfgs optimization algorithm.
##5) train.txt and test.txt
  this two file are train dataset and test dateset.
  
