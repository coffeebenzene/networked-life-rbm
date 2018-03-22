import numpy as np
import scipy.special
import projectLib as lib

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))

def sig(x):
    ### Question 2.1 ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    return scipy.special.expit(x)

def visibleToHiddenVec(v, w):
    ### Question 2.2 ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F
    h = np.zeros(w.shape[1])
    for k in range(K):
        vk = v[:,k] # column k of matrix v as a 1D array
        wk = w[:,:,k] # Matrix k of tensor w.
        h += np.dot(vk,wk) # element i of vk matches with row i of wk
    h = sig(h)
    return h

def hiddenToVisible(h, w):
    ### Question 2.3 ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #     is the number of movies the user has seen.
    #     Remember that we do not reconstruct movies that the user
    #     has not rated! (where reconstructing means getting a distribution
    #     over possible ratings).
    #     We only do so when we predict the rating a user would have given to a movie.
    v = np.zeros((w.shape[0], K))
    for k in range(K):
        wk = w[:,:,k] # Matrix k of tensor w.
        v[:,k] = np.dot(wk,h) # column j of wk matches with element j of h.
    v = sig(v)
    return v

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq):
    ### Question 2.4 ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    h_prob = visibleToHiddenVec(v, w)
    h_sample = sample(h_prob)
    wq_tensor = wq[np.newaxis,:,:] # m x F x K tensor (m=1, K=5)
    predicted_vq = hiddenToVisible(h_sample, wq_tensor) # Returns 1xK matrix
    predicted_vq = predicted_vq[0]
    return predicted_vq

def predictRatingMax(ratingDistribution):
    ### Question 2.5 ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    max_prob = np.max(ratingDistribution)
    likely_ratings = np.flatnonzero(ratingDistribution == max_prob)
    rating = np.median(likely_ratings) # In event of tie, choose middle rating
    return rating

def predictRatingMean(ratingDistribution):
    ### Question 2.5 ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over ratingDistribution
    rating_sum = np.dot(ratingDistribution, [1,2,3,4,5])
    prob_sum = np.sum(ratingDistribution)
    return rating_sum/prob_sum

def predictRatingExp(ratingDistribution):
    ### Question 2.5 ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over
    # the softmax applied to ratingDistribution
    softmax_distribution = softmax(ratingDistribution)
    expected_rating = np.dot(softmax_distribution, [1,2,3,4,5])
    return expected_rating

def predictMovieForUser(q, user, W, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    elif predictType == "mean":
        return predictRatingMean(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, training, predictType="exp"):
    ### Part 3.1 ###
    # given a user ID, predicts all movie ratings for the user
    user_ratings = np.zeros(W.shape[0])
    for q in range(len(user_ratings)):
        rating = predictMovieForUser(q, user, W, training)
        user_ratings[q] = rating
    return user_ratings
