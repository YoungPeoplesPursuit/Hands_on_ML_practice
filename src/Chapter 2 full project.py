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
print(housing.head()) #First five rows of dataset
print(housing.info()) #Displays data types and information like missing values, total values, etc
print(housing["ocean_proximity"].value_counts()) # for categorical data
print(housing.describe()) #general data like quartiles, max, min, mean, std

#histiograms to visualize data spread
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.savefig("Housing_attribute_histogram_plots.png")
#plt.show()

#################################TEST SET#################################################
from sklearn.model_selection import train_test_split

#Default test set
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(test_set.head())

#Test set representative of the median income distribution

#seperate median income into categories
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
print(housing["income_cat"].value_counts())
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]): #picks test data proportional to median income distribution
    strat_train_set = housing.loc[train_index] #training set
    strat_test_set = housing.loc[test_index] #testing set

#see how close the proportions are
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set): #we don't need the income categories anymore
    set_.drop("income_cat", axis=1, inplace=True)

#################################DATA VIS AND CLEAN#################################################
housing = strat_test_set.copy() #now work with a copy of the training data

#basic scatterplot that sort of looks like California
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")