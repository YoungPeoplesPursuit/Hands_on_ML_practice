import pandas as pd
import numpy as np
import sklearn

#Housing prices in California

############################BASIC DATA#####################################

#Download the data
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#Look at the data
housing = load_housing_data() #the main dataset
#print(housing.head()) #First five rows of dataset
#print(housing.info()) #Displays data types and information like missing values, total values, etc
#print(housing["ocean_proximity"].value_counts()) # for categorical data
#print(housing.describe()) #general data like quartiles, max, min, mean, std

#histiograms to visualize data spread
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.savefig("Housing_attribute_histogram_plots.png")
#plt.show()

#################################TEST SET#################################################
from sklearn.model_selection import train_test_split

#Default test set
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print(test_set.head())

#Test set representative of the median income distribution

#seperate median income into categories
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
#print(housing["income_cat"].value_counts())
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]): #picks test data proportional to median income distribution
    strat_train_set = housing.loc[train_index] #training set
    strat_test_set = housing.loc[test_index] #testing set

#see how close the proportions are
#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set): #we don't need the income categories anymore
    set_.drop("income_cat", axis=1, inplace=True)

#################################DATA VIS WITH CALIFNORNIA MAP#################################################
housing = strat_test_set.copy() #now work with a copy of the training data

#basic scatterplot that sort of looks like California
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
#parameters explanied: s is datapoint size. alpha is transparency. cmap is color scheme
plt.legend()
plt.savefig("housing_prices_scatterplot")
#plt.show()

#now putting the scatterplot on a map of California
'''
# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

#put the plot on a map of california
import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                  s=housing['population']/100, label="Population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

#customized tickmarks and labels for color bar
prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
#plt.show()
'''
####################################CORRELATIONS############################################
corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False)) #median house value has a decently high correlation with median income

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.savefig("income_vs_house_value_scatterplot")

#experimenting with other variables
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] #because multiple households in a district
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] #apparently bedroom/room ratio has a decent correlation with value
housing["population_per_household"]=housing["population"]/housing["households"] #rough size of household
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

#############################DATA CLEANING AND PREPARING FOR TRAINING#####################################
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

#Deal with missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") #inputs the median for missing values
housing_num = housing.drop("ocean_proximity", axis=1) #get rid of categorical data
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
#print(imputer.statistics_)

#check to see that the imputer worked

X = imputer.transform(housing_num) #transform the dataset
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
#print(housing_tr.loc[sample_incomplete_rows.index.values]) #print the incomplete rows with filled data


#Deal with categorical data
housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder() #turns categorical data into numerical data by assigning 1 or 0 for all categories to see which category it is in
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder() #makes an array out of the ordinal encoder
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(housing_cat_1hot) #returns a sparse array

housing_cat_1hot.toarray() #turn that into a normal dense array


###########################TRANSFORMERS############################

#Combine all of those new variables we made
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin): #custom transformer to add all the new columns
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room #because it had a somewhat high correlation with median home value
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X): #change the dataframe
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#Recover the column names and the dataframe. Numpy array doesn't have columns. Scikit converts to numpy arrays
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

#the pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), #impute missing values
        ('attribs_adder', CombinedAttributesAdder()), #add new attributes
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#encoding the categorical data into numbers
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num) #a list of the numerical variables
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([ #combining all of that
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing) #what we want for training
print(housing_prepared)

###############################MODELS TO TEST########################################

#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5] #real labels
some_data_prepared = full_pipeline.transform(some_data) #the data transformed and normalized

print("Predictions:", lin_reg.predict(some_data_prepared)) #run selected data through model

print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared) #run the data through the model
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse) #the error...
print(lin_rmse) #is HUGE!!!!!!

#Desicion tree
from sklearn.tree import DecisionTreeRegressor #try a new model

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse #yo what?!?! no error?!?! time to test

#cross validate desicion tree
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10) #10-fold splitting
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

#display_scores(tree_rmse_scores) #oh shoot thats huge!

#cross validate linreg
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
#display_scores(lin_rmse_scores) #somehow better than desicion tree!



#random forest does random desicion trees and averages those out
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

#cross validate the random forest
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores) #ok these get smaller

#############################FINE TUNING###########################################
#Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_ )

print(grid_search.best_estimator_)

#the scores of all the hyperparameter combinations!!!
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#put into a dataframe
pd.DataFrame(grid_search.cv_results_)


# randomized search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
   }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


#analyze best models and errors
feature_importances = grid_search.best_estimator_.feature_importances_ #which variables have the most weight?

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True) #importances of all variables
##############################TEST SET TIME##############################################
final_model = grid_search.best_estimator_ #testing the best model on the test set!

X_test = strat_test_set.drop("median_house_value", axis=1) #drop the house values labels for data to test!
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)