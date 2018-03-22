import numpy as np
import random

def getTrainingData():
    return np.genfromtxt("training.csv", delimiter=",", dtype=np.int)

def getValidationData():
    return np.genfromtxt("validation.csv", delimiter=",", dtype=np.int)

def getChapter4Data():
    ### MODIFIED FOR numpy arrays ###
    # chapter 4 training dataset
    return np.array([[0, 0, 5], [2, 0, 5], [3, 0, 4], [1, 1, 1], [2, 1, 1], 
                     [3, 1, 4], [0, 2, 4], [1, 2, 1], [2, 2, 2], [3, 2, 4],
                     [0, 3, 3], [1, 3, 4], [3, 3, 3], [0, 4, 1], [1, 4, 5],
                     [2, 4, 3]])

def getUsefulStats(training):
    ### MODIFIED FOR numpy arrays ###
    training = np.array(training)
    movies = training[:,0]
    u_movies = np.unique(movies)

    users = training[:,1]
    u_users = np.unique(users)

    return {
        "movies": movies, # movie IDs
        "u_movies": u_movies, # unique movie IDs
        "n_movies": len(u_movies), # number of unique movies

        "users": users, # user IDs
        "u_users": u_users, # unique user IDs
        "n_users": len(u_users), # number of unique users

        "ratings": training[:,2], # ratings
        "n_ratings": len(training), # number of ratings
    }

def getRatingsForUser(user, training):
    # user is a user ID
    # training is the training set
    # ret is a matrix, each row is [m, r] where
    #   m is the movie ID
    #   r is the rating, 1, 2, 3, 4 or 5
    return np.array([[x[0], x[2]] for x in training if x[1] == user])

# RMSE function to tune your algorithm
def rmse(r, r_hat):
    r = np.array(r)
    r_hat = np.array(r_hat)
    return np.linalg.norm(r - r_hat) / np.sqrt(len(r))
