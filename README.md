# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the data  
2. Use the Elbow Method to find the optimal number of clusters  
3. Apply K-Means clustering with the chosen number of clusters  
4. Assign and add cluster labels to the dataset  
5. Visualize the resulting clusters using a scatter plot

## Program:

Program to implement the K Means Clustering for Customer Segmentation.

Developed by: A Ahil Santo

RegisterNumber: 212224040018

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

x=data.iloc[:,3:]

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss,color="blue")
plt.title("Elbow Method")
plt.xlabel("No of Clusters")
plt.ylabel("WSCC")
plt.show()

km=KMeans(n_clusters=5)
km.fit(x)
pred=km.predict(x)
print("Prediction: ",pred)
data['cluster']=pred

colors=['red','blue','green','yellow','orange']

for i in range(5):
    cluster=data[data['cluster']==i]
    plt.scatter(cluster['Annual Income (k$)'],cluster['Spending Score (1-100)'],c=colors[i],label=f"Cluster {i}")

plt.legend()
plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

```

## Output:

Head

![1](https://github.com/user-attachments/assets/cb2a9574-56a9-4638-99be-e831337c3c96)

Info

![2](https://github.com/user-attachments/assets/00083bb2-78b7-4188-901d-a385243e4b31)

Null Values:

![3](https://github.com/user-attachments/assets/21f61412-8bbd-45b5-9360-ff5770af801f)

Elbow Method:

![4](https://github.com/user-attachments/assets/904d7a16-81e4-42a1-aa92-fa46b008819f)

Y_Prediction:

![5](https://github.com/user-attachments/assets/411414b4-c6bb-4a59-a2f0-c7105a3ec19d)

Customer Segments:

![6](https://github.com/user-attachments/assets/dc9705cc-38e8-43c1-bbf2-11c276a7e8eb)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
