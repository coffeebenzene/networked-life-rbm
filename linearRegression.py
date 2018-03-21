import numpy as np
import matplotlib.pyplot as plt

import projectLib

# Inputs should be numpy 2D array, with missing values as nan.
# Arrays will be converted into masked arrays.
# Rows are users, columns are movies.

class BaselinePredictor:
    def __init__(self, data, regulariser=0):
        """
        Takes in numpy data array, [[movie, user, rating] ...] as training data.
        Users correspond to rows, movies correspond to columns.
        Note: Regulariser will be scaled proportional to the number of data 
              elements by the class. i.e. regulariser used will be *n.
        Computation uses least square regularised version as in HW2 (for Q4)
        i.e. solve for b: (A.T*A + regulariser*n*I) b = A^T*c
        """
        self.train, self.stats = data_to_mat_stats(data)
        
        # c vector (baseline ratings)
        ratings = data[:,[2]] # [2] in column dimension to select column vector
        self.overall_mean = np.mean(ratings)
        c = ratings - self.overall_mean
        
        # A matrix
        num_elements = self.stats["n_ratings"]
        data_width = self.stats["n_users"] + self.stats["n_movies"]
        A = np.zeros((num_elements, data_width))
        for i, element in enumerate(data):
            movie, user, rating = element
            A[i, user] = 1
            A[i, self.stats["n_users"] + movie] = 1
        
        # compute left and right sides of regression equation
        left = np.dot(A.T,A) + num_elements*regulariser*np.eye(A.shape[1])
        right = np.dot(A.T, c)
        
        # Get (any) solution to regression
        linear_prog = np.linalg.lstsq(left, right)
        opt_parameters = linear_prog[0]
        
        # Extract biases
        self.row_bias = opt_parameters[:self.stats["n_users"]]
        self.col_bias = opt_parameters[self.stats["n_users"]:]
        
        # check: rank of data < number of columns (not full rank)
        if linear_prog[2] < data_width:
            print("WARNING: matrix is singular. rank {}<{}".format(linear_prog[2], data_width))
    
    def predict(self, row, col):
        val = self.overall_mean + self.row_bias[row] + self.col_bias[col]
        return val[0] # remove array.
    
    def predict_clipped(self, row, col):
        return np.clip(self.predict(row, col), 0, 5)
    
    def predict_complete(self):
        return self.overall_mean + self.row_bias + self.col_bias.T
    
    def predict_complete_clipped(self):
        return np.clip(self.predict_complete(), 0, 5)



def data_to_mat_stats(data):
    stats = projectLib.getUsefulStats(data)
    mat = np.empty((stats["n_users"], stats["n_movies"]))
    mat.fill(np.nan)
    for element in data:
        movie, user, rating = element
        mat[user, movie] = rating
    return mat, stats

def predicted_rmse(model, data):
    total_sq_error = 0
    num_elements = data.shape[0]
    for element in data:
        movie, user, rating = element
        prediction = model.predict_clipped(user, movie)
        sq_error = (prediction - rating)**2
        total_sq_error += sq_error
    rmse = np.sqrt(total_sq_error/num_elements)
    return rmse

def log_lambda_guess(model, train_data, validation_data, min_log_l, max_log_l, number=11):
    """
    Computes the rmse of model with a series of lamba/regulariser value.
    By applying this to validation data, the lambda which results in the 
    smallest rmse is the best regulariser.
    Since the range of lambdas can be very large, guessing is done over a log10 
    transformation.
    Instead of guessing many numbers over the range, a small number of intervals
    is guessed, then a smaller range can be obtained. Function can be iterated
    multiple times for convergence.
    
    Input is the min and max lambda in log10
    Output is a smaller min and max lambda in log 10
    """
    log_lambdas = np.linspace(min_log_l, max_log_l, number)
    lambdas = 10**log_lambdas
    rmse_list = []
    
    for l in lambdas:
        model = BaselinePredictor(train_data, regulariser=l)
        rmse = predicted_rmse(model, validation_data)
        rmse_list.append(rmse)
    # Find index of min RMSE
    index_min = np.argsort(rmse_list)[0]
    # Get new range of log lambda, done by bracketing 1 index left and right
    # from the initial range of log_lambdas
    if index_min-1 < 0:
        min_log_l2 = log_lambdas[0]
        print("Warning: Min RMSE occurs at left edge of range")
    else:
        min_log_l2 = log_lambdas[index_min-1]
    if index_min+1 >= number:
        max_log_l2 = log_lambdas[number]
        print("Warning: Min RMSE occurs at right edge of range")
    else:
        max_log_l2 = log_lambdas[index_min+1]
    return min_log_l2, max_log_l2



if __name__ == "__main__":
    np.set_printoptions(precision=3)
    
    # Question 1.1 - RMSE of training data. (Training loss)
    print("Non-regularised model")
    
    # train_data = np.array(projectLib.getChapter4Data())
    train_data = projectLib.getTrainingData()
    validation_data = projectLib.getValidationData()
    
    # Do prediction, mse is computed in the BaselinePredictor class
    model = BaselinePredictor(train_data)
    
    # Print RMSE.
    """
    print("Predicted complete matrix:")
    print(model.predict_complete())
    print("Overall mean:")
    print(model.overall_mean)
    print("Biases:")
    print(model.row_bias)
    print(model.col_bias.T)
    """
    rmse = predicted_rmse(model, train_data)
    print("training RMSE = {}".format(rmse))
    rmse = predicted_rmse(model, validation_data)
    print("validation RMSE = {}".format(rmse))
    
    
    # Question 1.2 Regularised model
    # Linear regression part of Question 3: Fine tuning regularisation parameter
    # Regularisation parameter is determined by guess and check.
    print("Regularised model")
    
    # Start with lambda in [-5,5]
    min_log_l = -5
    max_log_l = 5
    
    # Iterate guess and check 6 times (after testing, it converges after 6 times)
    for i in range(6):
        min_log_l, max_log_l = log_lambda_guess(model, train_data, validation_data, 
                                                min_log_l, max_log_l)
    log_l = (min_log_l + max_log_l)/2
    l = 10**log_l
    
    # Regularised model
    print("lambda = {}".format(l))
    model = BaselinePredictor(train_data, regulariser=l)
    
    rmse = predicted_rmse(model, train_data)
    print("training RMSE = {}".format(rmse))
    rmse = predicted_rmse(model, validation_data)
    print("validation RMSE = {}".format(rmse))
    
    # Write prediction
    print("Writing complete prediction...")
    np.savetxt("eric_kangraye_shaun+v1.txt", model.predict_complete())
    input("Press Enter to finish")