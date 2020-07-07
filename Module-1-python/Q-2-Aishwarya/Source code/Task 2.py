import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(Path('Downloads\\Customers.csv'))

print(data["CustomerID"].value_counts())
#nulls count
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)

data.loc[(data['Annual Income (k$)'].isnull()==True),'Annual Income (k$)']=data['Annual Income (k$)'].mean()
data.loc[(data['Spending Score (1-100)'].isnull()==True),'Spending Score (1-100)']=data['Spending Score (1-100)'].mean()

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
print(x.shape,y.shape)
from sklearn import metrics
score = metrics.silhouette_score(x, y)
print('The silhortte score before clustering',score)

#elbow method to know the number of clusters
wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,7),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#nclusters = 5 # this is the k in kmeans
km = KMeans(n_clusters=5)
km.fit(x)
y_cluster_kmeans= km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('The Silhoutte score after kmeans',score)
#clusters
x['results'] = y_cluster_kmeans
sns.FacetGrid(x, hue="results", height=4).map(plt.scatter, 'Annual Income (k$)', 'Spending Score (1-100)').add_legend()
plt.title("X CLUSTERED")
plt.show()





