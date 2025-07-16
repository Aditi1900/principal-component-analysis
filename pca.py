from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pit

df = pd.read_csv("Iris.csv")

print(df.head())

labels = df['Species']

X = df.drop(["Id", "Species"], axis=1)
# We are getting rid of the Id and Species col, and as they are cols we are using axis=1
# X is the entire dataframe minus the Id and the Species

# Step 1 of PCA - Center all xi around the origin

# Create an object for standard scalar and then use fit transform to center the Xs around the origin
X_std = StandardScaler().fit_transform(X)

# Initialize pca by calling the PCA method from scikit learN
# Here we are just trying to see how many components are actually useful
# As creating 4 components doesn't make sense as we already have 4 features i.e. cols in the table i.e. SepalLengthCm etc.
pca = PCA(n_components=2)

X_transform = pca.fit_transform(X_std)

print("\n")
print(pca.explained_variance_)
# these values are not in terms of percentages.

print(pca.explained_variance_ratio_)
# This means that in the first principal component we are capturing 72 percent of the variance, in the second one 23 percent and so on
# Looking at the 4 numbers, it's better to retain just the first 2 principal components as they are capturing 95% of the variance

print(X_transform)
# Prints out the two cols we want to extract

pca1 = list(zip(*X_transform))[0]
# Earlier it was pca1 = zip(*X_transform)[0], had to change cause I got an error, apparantly, In Python 3, zip() returns an iterator,
# tggnot a list. Iterators are not subscriptable, which means you can't do [0] directly on them like a list.
# Inverse zip of X transform and get the 0th element of that list i.e. the first col

pca2 = list(zip(*X_transform))[1]
# Earlier it was pca2 = zip(*X_transform)[1]
# Same as above for getting the 2nd col

#Creating a colors dictionary
color_dict = {}
color_dict["Iris-setosa"] = "green"
color_dict["Iris-versicolor"] = "red"
color_dict["Iris-virginica"] = "blue"

# To visualize it, we plot all the points in col 1 and 2
i=0
for label in labels:
    plt.scatter(pca1[i], pca2[i], color=color_dict[label])
    i=i+1

plt.show()