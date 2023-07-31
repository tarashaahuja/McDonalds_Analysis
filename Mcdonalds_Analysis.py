#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Collecting The Data

# In[56]:


df=pd.read_csv("/Users/TT/Desktop/mcdonalds.csv")
df


# Exploring the Data

# In[57]:


df.shape
df.dtypes
df.info()
df.isnull().sum()


# In[58]:


df.columns


# In[59]:


df.shape


# In[60]:


col=df.columns
col=col[:11]
bf=df
for i in col:
  bf[i]=bf[i].map({'Yes':1,'No':0})
bf


# Label encoding the orginal data set

# In[61]:


#Label encoding for categorical - Converting 11 cols with yes/no

from sklearn.preprocessing import LabelEncoder
def labelling(x):
    df[x] = LabelEncoder().fit_transform(df[x])
    return df

cat = ['yummy','convenient','spicy','fattening','greasy','fast','cheap','tasty','expensive','healthy','disgusting']

for i in cat:
    labelling(i)
df


# In[62]:


bf=bf.iloc[:,:11]


# In[63]:


bf.describe()


# In[64]:


bf.corr()


# In[65]:


plt.figure(figsize=(7,7))
sns.heatmap(bf.corr(), annot=True)
plt.show()


# In[66]:


df.query('Gender == "Male"').Gender.count()


# In[67]:


df.query('Gender == "Female"').Gender.count()


# In[68]:


labels = ['Male','Female']
sizes = [df.query('Gender == "Male"').Gender.count(),df.query('Gender == "Female"').Gender.count()]
#colors
colors = ['red','blue']
#explsion
explode = (0.01,0.01)
plt.figure(figsize=(8,8))
my_circle=plt.Circle( (0,0), 0.5, color='black')
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.80,explode=explode)
p=plt.gcf()
plt.axis('equal')
p.gca().add_artist(my_circle)
plt.show()


# Perfroming ELBOW Method to find out the most suitable value for k

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[20]:


X=bf
mapping1 = {}
K = range(1, 10)
t=[]

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    t.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)))
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]


# In[21]:


for key, val in mapping1.items():
    print(f'{key} : {val}')


# Plotting the histogram based on the sum within the cluster distances

# In[22]:


fig = plt.figure(figsize =(10,7))
plt.xlabel("number of segments")
plt.ylabel("sum of within cluster distances")
# Horizontal Bar Plot
plt.bar(K,t)

# Show Plot
plt.show()


# Through the above graph we can say that the the sharp edge of the elbow is at no. of segment = 5, Thus further we will use k=5

# Reducing the dimension using PCA so that we can easily visualize it

# In[23]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(X)

pca = PCA(n_components=11)
pc = pca.fit_transform(X)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pf = pd.DataFrame(data = pc, columns = names)
pf


# In[24]:


pca.explained_variance_ratio_


# In[25]:


np.cumsum(pca.explained_variance_ratio_)


# In[26]:


loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = bf.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[27]:


plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# Extracting the Segments

# Performing K- means Clustering and visualising it over pca results as x and y label

# In[28]:


# Perform clustering 
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(bf)

# Plot the PCA results with different colors for each cluster
plt.scatter(pc[:, 0], pc[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA with Clustering')
plt.colorbar(label='Cluster')
plt.show()

# Display projection axes (principal components)
print("PCA Projection Axes:")
print(pca.components_)


# Performing Hierarchial Clustering

# In[29]:


from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering


# In[30]:


merg = linkage(bf,method="ward")
plt.figure(figsize=(25,10))
dendrogram(merg,leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# In[31]:


#create model
hiyerartical_cluster = AgglomerativeClustering(n_clusters = 5,affinity= "euclidean",linkage = "ward")
data_predict = hiyerartical_cluster.fit_predict(bf)
plt.figure(figsize=(8,8))
plt.scatter( pc[:, 0], pc[:, 1] , c = data_predict , s = 100 )
plt.show()


# Now working over the LIKE, VisitFrequency Columns

# In[32]:


#Customer segmentation - based on pyschographic segmentation

#For convinence renaming the category
df['Like']= df['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
#Like
sns.catplot(x="Like", y="Age",data=df,orient="v", height=5, aspect=2, palette="Set2",kind="swarm")
plt.title('Likelyness of McDonald w.r.t Age')
plt.show()


# In[33]:


#Histogram of the each attributes
plt.rcParams['figure.figsize'] = (12,14)
df.hist()
plt.show()


# In[35]:


import bioinfokit


# In[36]:


from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[37]:


pca_scores = PCA().fit_transform(X)
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))


# In[38]:


df['cluster_num'] = kmeans.labels_#adding to df


# In[39]:


df


# In[40]:


#DESCRIBING SEGMENTS

from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

crosstab =pd.crosstab(df['cluster_num'],df['Like'])
#Reordering cols
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab


# In[41]:


#MOSAIC PLOT
plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab.stack())
plt.show()


# In[42]:


#Mosaic plot gender vs segment
crosstab_gender =pd.crosstab(df['cluster_num'],df['Gender'])
crosstab_gender


# Represents ratio of Male and Female in every cluster

# In[43]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[44]:


#box plot for age

sns.boxplot(x="cluster_num", y="Age", data=df)


# In[45]:


# Selecting target segment

#Calculating the mean
#Visit frequency
df['VisitFrequency'] = LabelEncoder().fit_transform(df['VisitFrequency'])
visit = df.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[46]:


#Like
df['Like'] = LabelEncoder().fit_transform(df['Like'])
Like = df.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[47]:


#Gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
Gender = df.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[48]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[49]:


#Target segments

plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="orange")
plt.title("Simple segment evaluation plot for the fast food data set",fontsize = 15)
plt.xlabel("Visit", fontsize = 12)
plt.ylabel("Like", fontsize = 12)
plt.show()


# In[ ]:




