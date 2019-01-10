"""
Customer Segmentation@Bambu.life v.0.1.5 (2018)
====================================================================================================
Author: Tay Paul Hong<paul@bambu.life>

Scope: The algo uses Clustering (Unsupervised learning) to group people into different clusters

General Description:
---------------------
This model helps clients to determine the number of customer segments they need, and the parameters for each segment.
Once these segments are determined, the right marketing strategy and user journey can be created to improve the user exerpeince.

New in This version:
---------------------
Add in sklearn.metrics.silhouette_score to see how accurate the model is

Current Issues/To do:
--------------------
"""
#-------------------------------------Initialization---------------------------------------------------

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#using OneHotEncoder to add categorical input
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

#Select working directory
import os
os.chdir('/Users/paul/Dropbox/personal stuff/SMU MITB')

#----------------------------Data import and preprocessing------------------------------------

#Importing user data that passed the health check. The dataset is assumed in .xls format;generalization to other formats is
#implemented using the correct pandas import function.

pp = pd.read_excel('NewDummyData_All.xlsx', 0, skiprows=1) # import excel spreadsheet, first layer (0)
pp.columns #Entries in the dataframe
pp.head() #plot the first 5 entries in the dataframe
pp['Age']#get the column 'Age'



#gathering the catagorial input into dataframe pp2
# numerical_v = input('Enter Numerical Variables with"/"').strip('').split('/')
numerical_v = ['Age','Monthly Income']
column=list(pp.columns)
#use subset to check if numerical_v is included in the pp dataframe
if set(numerical_v)<set(column):
    numerical_v=numerical_v
    pp1=pp[numerical_v]
else:
    print('the numerical input is not valid')

#gathering the catagorial input into dataframe pp2
# catagorial_v = input('Enter Catagory Variables with"/"').strip('').split('/')
catagorial_v = ['']
#use subset to check if catagorial_v is included in the pp dataframe, if it is, include it in the df, it not, ignore it.

if set(catagorial_v)<set(column):
    catagorial_v=catagorial_v
    nationality_1=pp[catagorial_v]
    nationality_1 = LabelEncoder().fit_transform(nationality_1)
    nationality_n=OneHotEncoder(sparse=False).fit_transform(nationality_1.reshape(-1,1))
    pp2 = pd.DataFrame(nationality_n)
    df = pd.concat([pp1,pp2],axis=1)
else:
    print('the catagoriall input is not valid')
    df = pp1


#---------------------------------------------Preprocessing-------------------------------------------------------

#replace NaN values with the mean
from sklearn.preprocessing import Imputer
im = Imputer(missing_values=np.nan, copy=True, strategy='mean')
X = im.fit_transform(df)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Scaled = sc_X.fit_transform(X)

#-----------------------------------Running the Hirachical Clustering Model----------------------------------------------------
# Using the dendrogram to find the optimal number of clusters
# Reccomended number of clusters is determined by the colors in the dendrogram
plt.figure(1)

#Select the method to optimize (Either ward, average, complete)
# method=input('Enter Linkage method (ward/average/complete):')
method = 'ward' # ward/average/complete
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X_Scaled, method = method))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

plt.savefig('Dendrogram.png')

# Fitting Hierarchical Clustering to the dataset
plt.figure(2)
no_of_cluster = 3
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = no_of_cluster, affinity = 'euclidean', linkage = method)
y_hc = hc.fit_predict(X_Scaled)


#-----------------------------------Visualising the clusters----------------------------------------------------

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 20, c = '#FF0000', marker='D')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 20, c = '#FFAC33',marker='D')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 20, c = '#CA33FF',marker='D')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 20, c = '#33FF77',marker='D')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 20, c = '#33FFD1',marker='D')
plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 20, c = '#33BEFF', marker='D')
plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 20, c = '#3355FF',marker='D')
plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 20, c = '#7733FF',marker='D')
plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s = 20, c = '#F9FF33',marker='D')
plt.scatter(X[y_hc == 9, 0], X[y_hc == 9, 1], s = 20, c = '#FF339C',marker='D')


plt.title('Clusters of customers')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()

#Save the chart
plt.savefig('Scatter.png')

#Showing the silhouette score
from sklearn.metrics import silhouette_score
score=silhouette_score(X_Scaled, y_hc)
print ('Silhouette score =',score)

#Validating the model 

ClusterSizes=np.arange(2,6)
linkage=['ward','average','complete']
affinity=['euclidean',]#, 'l1', 'l2', 'manhattan', 'cosine']
for i in range(len(ClusterSizes)):
    for j in range(len(linkage)):
        for k in range(len(affinity)):
            hc = AgglomerativeClustering(n_clusters = ClusterSizes[i], affinity = affinity[k], linkage = linkage[j])
            y_hc = hc.fit_predict(X_Scaled)
            score=silhouette_score(X_Scaled, y_hc)
            print('Cluster size=', ClusterSizes[i],', linkage=',linkage[j],', affinity=', affinity[k], ', score=',score)
