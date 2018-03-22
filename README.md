# networked-life-rbm
01.104 Networked Life - build a Restricted Boltzmann Machine to do the netflix problem

1001526 Eric Teo  
1001531 Shaun Yee  
1001454 Ng Kang Raye  

## Note about modification of given template code

The given template code (stored in studentcode) mixes python lists and numpy arrays.
This has been modified to solely use numpy arrays where meaningful.
This ensures more robust code, and also better performance.

Also, the original code seems to be for python 2. It has also been modified for python 3, since our group uses python 3.

Some bugs observed were also fixed.

These changes are noted in the code via comments, in the from of `### MODIFIED FOR <reason> ###`.

## LinearRegression.py

LinearRegression.py was completely rewritten based on the group's HW4.

The prediction model is put into a class for better usability.

Results:

Non-regularised model  
training RMSE = 0.8470026505119093  
validation RMSE = 1.069395279922955

Regularised model  
lambda = 0.0027179408435172364  
training RMSE = 0.8981572110536863  
validation RMSE = 1.0195162850501946