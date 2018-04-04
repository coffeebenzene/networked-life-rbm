import numpy as np
import matplotlib.pyplot as plt

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
        i.e. solve for b: (A.T*A + n*regulariser*I) b = A^T*c
        """
        n_users = np.max(data[:,1])+1 # Assume no "holes" (unassigned values)
        n_movies = np.max(data[:,0])+1
        
        # c vector (baseline ratings)
        ratings = data[:,[2]] # [2] in column dimension to select column vector
        self.overall_mean = np.mean(ratings)
        c = ratings - self.overall_mean
        
        # A matrix
        num_elements = len(data)
        data_width = n_users + n_movies
        A = np.zeros((num_elements, data_width))
        for i, element in enumerate(data):
            movie, user, rating = element
            A[i, user] = 1
            A[i, n_users + movie] = 1
        
        # compute left and right sides of regression equation
        # left = (A.T*A + n*regulariser*I)
        # right = A^T*c
        left = np.dot(A.T,A) + num_elements*regulariser*np.eye(A.shape[1])
        right = np.dot(A.T, c)
        
        # Get (any) solution to regression
        linear_prog = np.linalg.lstsq(left, right)
        opt_parameters = linear_prog[0]
        
        # Extract biases
        self.row_bias = opt_parameters[:n_users]
        self.col_bias = opt_parameters[n_users:]
        
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

class kNNPredictor:
    def __init__(self, data, regulariser=0, krow=0, kcol=0):
        """
        Takes in numpy data array, [[movie, user, rating] ...] as training data.
        Users correspond to rows, movies correspond to columns.
        Computes via BaselinePredictor first, then does kNN.
        Copied from HW2
        """
        self.baseline = BaselinePredictor(data, regulariser)
        self.train = data_to_ma(data)
        self.train += 0.01  # Offset to prevent division by 0 for cos similarity
                            # Prediction is linear, so shouldn't affect results.
        self.error_mat = self.train - self.baseline.predict_complete_clipped()
        
        nrows, ncols = self.train.shape
        
        # Compute column similarity matrix, and array of k nearest columns.
        col_sim = np.ma.zeros((ncols, ncols))
        col_sim[np.diag_indices(ncols)] = np.ma.masked  # Mask same value (i,i) pairs.
        for i,j in zip(*np.tril_indices(ncols, k=-1)):  # get unique pairs
            col_i = self.error_mat[:,i]
            col_j = self.error_mat[:,j]
            col_sim[i,j] = cos_similarity(col_i, col_j)  # populate similarity matrix
            col_sim[j,i] = col_sim[i,j]
        # absolute value to account for inverse correlation
        # negate so that sort will put larger correlation first (numpy sorts small->big)
        col_abs_sim = -np.abs(col_sim)
        # Sort and extract first kcol neighbour columns.
        # col i in train matrix corresponds to row i in neighbour_cols.
        self.neighbour_cols = np.argsort(col_abs_sim)[:,:kcol]
        self.col_sim = col_sim
        
        # Compute row similarity matrix, and array of k nearest rows.
        row_sim = np.ma.zeros((nrows, nrows))
        row_sim[np.diag_indices(nrows)] = np.ma.masked  # Mask same value (i,i) pairs.
        for i,j in zip(*np.tril_indices(nrows, k=-1)):  # get unique pairs
            row_i = self.error_mat[i,:]
            row_j = self.error_mat[j,:]
            row_sim[i,j] = cos_similarity(row_i, row_j)  # populate similarity matrix
            row_sim[j,i] = row_sim[i,j]
        # Just like above, use negative absolute value for ordering.
        row_abs_sim = -np.abs(row_sim)
        # Sort and extract first krow neighbour rows.
        # row i in train matrix corresponds to row i in neighbour_rows.
        self.neighbour_rows = np.argsort(row_abs_sim)[:,:krow]
        self.row_sim = row_sim
    
    def predict(self, row, col):
        val = self.baseline.predict(row, col)
        
        # Compute for column similarity
        neighbour_cols = self.neighbour_cols[col]
        errors_col = self.error_mat[row, neighbour_cols] # error for row, across columns.
        col_sim = np.ma.MaskedArray(self.col_sim[col, neighbour_cols], errors_col.mask)
        
        weighted_sum = np.sum(col_sim*errors_col) 
        weights = np.sum(np.abs(col_sim))
        
        if (not np.ma.is_masked(weights)) and weights != 0:
            val += weighted_sum/weights
        
        # Compute for row similarity
        neighbour_rows = self.neighbour_rows[row]
        errors_row = self.error_mat[neighbour_rows, col] # error for column, down rows.
        row_sim = np.ma.MaskedArray(self.row_sim[row, neighbour_rows], errors_row.mask)
        
        weighted_sum = np.sum(row_sim*errors_row) 
        weights = np.sum(np.abs(row_sim))
        
        if (not np.ma.is_masked(weights)) and weights != 0:
            val += weighted_sum/weights
        
        return val
    
    def predict_clipped(self, row, col):
        return np.clip(self.predict(row, col), 0, 5)
    
    def predict_complete(self):
        predicted = np.empty(self.train.shape)
        for row, col in np.ndindex(predicted.shape):
            predicted[row, col] = self.predict(row, col)
        return predicted
    
    def predict_complete_clipped(self):
        return np.clip(self.predict_complete(), 0, 5)

def cos_similarity(u, v):
    # Compute mask
    or_mask = np.logical_or(u.mask, v.mask)
    # Reject if only 1 common value. (Too little to correlate)
    if np.sum(np.logical_not(or_mask)) <= 1:
        return 0
    # Handle masking
    u = np.ma.MaskedArray(u, or_mask)
    v = np.ma.MaskedArray(v, or_mask)
    # based on scipy.spatial.distance.cosine (which calls .correlation)
    uv = np.asscalar(np.sum(u*v))
    uu = np.sum(np.square(u))
    vv = np.sum(np.square(v))
    return uv / np.sqrt(uu * vv)

def data_to_ma(data):
    """ converts sequence of points [col, row, value] to numpy masked array
    """
    n_row = np.max(data[:,1])+1 # Assume no "holes" (unassigned values)
    n_col = np.max(data[:,0])+1
    mat = np.ma.masked_all((n_row, n_col))
    for element in data:
        movie, user, rating = element
        mat[user, movie] = rating
    return mat

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

def log_lambda_guess(model_class, train_data, validation_data, min_log_l, max_log_l, number=11):
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
        model = model_class(train_data, regulariser=l)
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
    print("-"*40)
    print("Non-regularised model")
    
    train_data = np.genfromtxt("training.csv", delimiter=",", dtype=np.int)
    validation_data = np.genfromtxt("validation.csv", delimiter=",", dtype=np.int)
    #import projectLib
    #train_data = projectLib.getChapter4Data()
    #validation_data = projectLib.getChapter4Data()
    
    # Do prediction, mse is computed in the BaselinePredictor class
    model = BaselinePredictor(train_data)
    
    # Print RMSE.
    """
    print("Predicted complete matrix:")
    print(model.predict_complete_clipped())
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
    print()
    print("-"*40)
    print("Regularised model")
    
    l=0.0008796704432324492
    krow=0
    kcol=0
    """ 
    ## lambda determined to be 0.0008796704432324492 by guess and check.
    # Start with lambda in [-5,5]
    min_log_l = -5
    max_log_l = 5
    
    # Iterate guess and check 6 times (after testing, it converges after 6 times)
    for i in range(6):
        min_log_l, max_log_l = log_lambda_guess(BaselinePredictor, train_data, validation_data, 
                                                min_log_l, max_log_l)
        print("log_l range {} to {}".format(min_log_l, max_log_l))
    log_l = (min_log_l + max_log_l)/2
    l = 10**log_l
    print("l = {}".format(l))
    
    ## kcol, krow determined to be 0,0 by guess and check
    krow = 0
    kcol = 0
    lowest_error = None
    for i in range(6):
        for j in range(6):
            model = kNNPredictor(train_data, regulariser=l, krow=i, kcol=j)
            error = predicted_rmse(model, validation_data)
            if lowest_error is None or error < lowest_error:
                krow = i
                kcol = j
                lowest_error = error
            print("i,j={},{}, error={}, lowest_error={}, krow, kcol={}, {}".format(i,j, error, lowest_error, krow, kcol))
    """
    # for submission
    print("-"*40)
    print("lambda = {}".format(l))
    print("krow = {}".format(krow))
    print("kcol = {}".format(kcol))
    
    # Regularised model for Baseline
    print("-"*40)
    print("Baseline Model:")
    model = BaselinePredictor(train_data, regulariser=l)
    
    rmse = predicted_rmse(model, train_data)
    print("training RMSE = {}".format(rmse))
    rmse = predicted_rmse(model, validation_data)
    print("validation RMSE = {}".format(rmse))
    
    # Regularised model for kNN
    print("-"*40)
    print("kNN Model:")
    model = kNNPredictor(train_data, regulariser=l, krow=krow, kcol=kcol)
    
    rmse = predicted_rmse(model, train_data)
    print("training RMSE = {}".format(rmse))
    rmse = predicted_rmse(model, validation_data)
    print("validation RMSE = {}".format(rmse))
    
    # Combined prediction
    print("-"*40)
    train_data = np.concatenate((train_data,validation_data), axis=0)
    model = kNNPredictor(train_data, regulariser=l, krow=krow, kcol=kcol)
    
    # Write prediction
    print("Writing complete prediction...")
    np.savetxt("eric_kangraye_shaun+v2.txt", model.predict_complete_clipped())
    input("Press Enter to finish")