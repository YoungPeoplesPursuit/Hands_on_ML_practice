import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#model based learning with linear regression to predict happiness given GDP per capita

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values #returns the values of the GDPs and whatnot per country
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction") #plot the scatterpoint
plt.axis([23_500, 62_500, 4, 9]) #sets limits on the axis [x_min, x_max, y_min, y_max]
plt.show() #show the plot

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y) #passes batches of data through and calculates loss to train the model

# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new)) # outputs [[6.30165767]], which is its predicted happiness based on the model




#Instance based learning with nearest neighbors alg:

# Select a 3-Nearest Neighbors regression model
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
print(model.predict(X_new)) # outputs [[6.33333333]]

'''
How model.fit() works:

# Fake data: y = 4 + 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)   # 100 points, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (column of 1s)
X_b = np.c_[np.ones((100, 1)), X]  

# Init weights randomly (w0 = bias, w1 = slope)
theta = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

for iteration in range(n_iterations):
    # Predictions
    y_pred = X_b.dot(theta) #generate random weights
    
    # Gradient of MSE wrt theta
    gradients = 2/m * X_b.T.dot(y_pred - y) 
    
    # Update rule to get closer to the minimum 
    theta -= learning_rate * gradients 

print("Learned parameters (bias, slope):", theta.ravel())

'''
