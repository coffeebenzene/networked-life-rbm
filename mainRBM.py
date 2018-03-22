import numpy as np
import rbm
import projectLib as lib

import time

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!
# F = number of hidden units
F = 30
epochs = 10
gradientLearningRate = 0.1

# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
### MODIFIED FOR bugfix, gradient not reset for each gradient batch. ###
#posprods = np.zeros(W.shape)
#negprods = np.zeros(W.shape)

earlystop = np.copy(W)
earlystop_rmse = np.float("inf")

start = time.time()

for epoch in range(1, epochs+1): ### MODIFIED FOR bugfix, off-by-one error ###
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)
    
    for user in visitingOrder:
        ### MODIFIED FOR bugfix, gradient not reset for each gradient batch. ###
        posprods = np.zeros(W.shape)
        negprods = np.zeros(W.shape)
        # get the ratings of that user
        ratingsForUser = lib.getRatingsForUser(user, training)
        
        # build the visible input
        v = rbm.getV(ratingsForUser)
        
        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :]
        
        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        grad = gradientLearningRate * (posprods - negprods)

        W += grad
    
    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training, predictType="exp")
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training, predictType="exp")
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
    
    if vlRMSE < earlystop_rmse:
        earlystop = np.copy(W)
        earlystop_rmse = vlRMSE
    
    ### MODIFIED FOR python 3 print ###
    print("### EPOCH {} ###".format(epoch))
    print("Time = {}".format(time.time() - start))
    print("Training loss = {}".format(trRMSE))
    print("Validation loss = {}".format(vlRMSE))

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
print("Time = {}".format(time.time() - start))
np.savetxt("eric_kangraye_shaun+v2.txt", predictedRatings)
