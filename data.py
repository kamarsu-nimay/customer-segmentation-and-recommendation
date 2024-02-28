#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, average_precision_score


# In[57]:


# Generate demographic information
customer_ids = range(1, 5001)
ages = [random.randint(18, 75) for _ in customer_ids]
genders = [random.choice(['Male', 'Female']) for _ in customer_ids]
locations = [random.choice(['North America', 'South America', 'Africa', 'Australia', 'Europe', 'Asia']) for _ in customer_ids]
incomes = [round(random.uniform(20000, 150000), 2) for _ in customer_ids]


# Generate purchase history
purchase_dates = [datetime(2023, random.randint(1, 12), random.randint(1, 28)) for _ in range(5000)]
transactions = range(1, len(purchase_dates) + 1)
product_ids = [random.randint(1, 150) for _ in transactions]
purchase_amounts = [round(random.uniform(10, 500), 2) for _ in transactions]
quantities = [random.randint(1, 8) for _ in transactions]
customer_ids_purchases = random.choices(customer_ids, k=len(transactions))

purchase_data = pd.DataFrame({
    'Transaction_ID': transactions,
    'Customer_ID': customer_ids_purchases,
    'Product_ID': product_ids,
    'Purchase_Date': purchase_dates,
    'Purchase_Amount': purchase_amounts,
    'Quantity': quantities,
    'Gender': genders
})

# Generate browsing behavior
pages_visited = [random.randint(1, 20) for _ in customer_ids]
time_spent = [round(random.uniform(1, 60), 2) for _ in customer_ids]

browsing_data = pd.DataFrame({
    'Customer_ID': customer_ids,
    'Pages_Visited': pages_visited,
    'Time_Spent': time_spent
})

# Generate product interactions
product_ratings = [random.randint(1, 5) for _ in range(5000)]
customer_ids_reviews = random.choices(customer_ids, k=len(product_ratings))
product_ids_reviews = range(1, len(product_ratings) + 1)
reviews = ['This product is great!' if rating > 3 else 'This product needs improvement.' for rating in product_ratings]

product_interactions_data = pd.DataFrame({
    'Customer_ID': customer_ids_reviews,
    'Product_ID': product_ids_reviews,
    'Rating': product_ratings,
    'Review': reviews
})


# In[58]:


# Save data to CSV files
purchase_data.to_csv('purchase_history.csv', index=False)
browsing_data.to_csv('browsing_behavior.csv', index=False)
product_interactions_data.to_csv('product_interactions.csv', index=False)


# In[59]:


# Load the datasets
purchase_data = pd.read_csv('purchase_history.csv')
browsing_data = pd.read_csv('browsing_behavior.csv')
product_interactions_data = pd.read_csv('product_interactions.csv')


# In[60]:


purchase_data.head()


# In[61]:


browsing_data.head()


# In[62]:


product_interactions_data.head()


# In[64]:


# Check for missing values in each dataset
print("\nMissing Values in Purchase Data:")
print(purchase_data.isnull().sum())
print("\nMissing Values in Browsing Data:")
print(browsing_data.isnull().sum())
print("\nMissing Values in Product Interactions Data:")
print(product_interactions_data.isnull().sum())


# In[65]:


# Check for outliers or unusual values
print("\nPurchase Data Summary:")
purchase_data.describe()


# In[66]:


print("\nBrowsing Data Summary:")
browsing_data.describe()


# In[67]:


print("\nProduct Interactions Data Summary:")
product_interactions_data.describe()


# #  Feature Engineering

# In[68]:


# Purchase Data
# Feature Engineering
purchase_data['Total_Purchase_Amount'] = purchase_data['Purchase_Amount'] * purchase_data['Quantity']
purchase_data['Average_Purchase_Amount'] = purchase_data['Total_Purchase_Amount'] / purchase_data['Quantity']

# Handling missing values (if any)
purchase_data.fillna(0, inplace=True)  # Fill missing values with 0, you may choose a different strategy

# Scaling numerical features
scaler = StandardScaler()
purchase_data[['Purchase_Amount', 'Quantity', 'Total_Purchase_Amount', 'Average_Purchase_Amount']] = scaler.fit_transform(
    purchase_data[['Purchase_Amount', 'Quantity', 'Total_Purchase_Amount', 'Average_Purchase_Amount']])

print("Preprocessed Purchase Data:")
purchase_data.head()


# In[69]:


# Browsing Data
# Handling missing values (if any)
browsing_data.fillna(0, inplace=True)  # Fill missing values with 0, you may choose a different strategy

# Scaling numerical features
browsing_data[['Pages_Visited', 'Time_Spent']] = scaler.fit_transform(browsing_data[['Pages_Visited', 'Time_Spent']])

print("\nPreprocessed Browsing Data:")
browsing_data.head()


# In[75]:


# Product Interactions Data
# Feature Engineering
product_interactions_data['Total_Reviews'] = product_interactions_data.groupby('Customer_ID')['Product_ID'].transform('count')
product_interactions_data['Average_Rating'] = product_interactions_data.groupby('Customer_ID')['Rating'].transform('mean')

# Handling missing values (if any)
product_interactions_data.fillna(0, inplace=True)  # Fill missing values with 0, you may choose a different strategy

# Encoding categorical feature "Review" using one-hot encoding
product_interactions_data = pd.get_dummies(product_interactions_data, columns=['Review'])

print("\nPreprocessed Product Interactions Data:")
print(product_interactions_data.head())


# # Purchase Data

# In[76]:


# Distribution of purchase amounts
plt.figure(figsize=(10, 6))
sns.histplot(purchase_data['Purchase_Amount'], bins=20, kde=True)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.show()


# In[77]:


# Distribution of quantities
plt.figure(figsize=(10, 6))
sns.histplot(purchase_data['Quantity'], bins=20, kde=True)
plt.title('Distribution of Quantities')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()


# In[78]:


# Distribution of total purchase amounts
plt.figure(figsize=(10, 6))
sns.histplot(purchase_data['Total_Purchase_Amount'], bins=20, kde=True)
plt.title('Distribution of Total Purchase Amounts')
plt.xlabel('Total Purchase Amount')
plt.ylabel('Frequency')
plt.show()


# In[79]:


# Relationship between purchase amount and quantity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Purchase_Amount', y='Quantity', data=purchase_data)
plt.title('Relationship between Purchase Amount and Quantity')
plt.xlabel('Purchase Amount')
plt.ylabel('Quantity')
plt.show()


# # Browsing Data

# In[80]:


# Distribution of pages visited
plt.figure(figsize=(10, 6))
sns.histplot(browsing_data['Pages_Visited'], bins=20, kde=True)
plt.title('Distribution of Pages Visited')
plt.xlabel('Pages Visited')
plt.ylabel('Frequency')
plt.show()


# In[81]:


# Distribution of time spent
plt.figure(figsize=(10, 6))
sns.histplot(browsing_data['Time_Spent'], bins=20, kde=True)
plt.title('Distribution of Time Spent')
plt.xlabel('Time Spent')
plt.ylabel('Frequency')
plt.show()


# In[82]:


# Relationship between pages visited and time spent
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pages_Visited', y='Time_Spent', data=browsing_data)
plt.title('Relationship between Pages Visited and Time Spent')
plt.xlabel('Pages Visited')
plt.ylabel('Time Spent')
plt.show()


# # Product Interactions Data

# In[83]:


# Distribution of product ratings
plt.figure(figsize=(10, 6))
sns.histplot(product_interactions_data['Average_Rating'], bins=5, kde=True)
plt.title('Distribution of Product Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()


# In[84]:


# Distribution of total reviews
plt.figure(figsize=(10, 6))
sns.histplot(product_interactions_data['Total_Reviews'], bins=20, kde=True)
plt.title('Distribution of Total Reviews')
plt.xlabel('Total Reviews')
plt.ylabel('Frequency')
plt.show()


# In[85]:


# Relationship between average rating and total reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Rating', y='Total_Reviews', data=product_interactions_data)
plt.title('Relationship between Average Rating and Total Reviews')
plt.xlabel('Average Rating')
plt.ylabel('Total Reviews')
plt.show()


# # K-mean clustering for customer segmentation 

# In[86]:


# Combine relevant features for clustering
X = purchase_data[['Total_Purchase_Amount', 'Average_Purchase_Amount']].values

# Determine the optimal number of clusters using the elbow method
inertia_values = []
silhouette_scores = []
max_clusters = 10  # Maximum number of clusters to consider
for i in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))


# In[87]:


# Plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# In[88]:


# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[89]:


# Based on the elbow method and silhouette score, choose the optimal number of clusters
optimal_clusters = 3  # Adjust as needed

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X)

# Assign each customer to a segment based on their cluster membership
purchase_data['Segment_Kmeans'] = kmeans.labels_

# Display the segment assignments
print("Segment Assignments (K-means):")
purchase_data[['Customer_ID', 'Segment_Kmeans']].head()


# # Hierarchical Clustering for customer segmentation

# In[90]:


# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.show()


# In[91]:


# Choose the optimal number of clusters based on dendrogram
optimal_clusters_hierarchical = 3  # Adjust as needed

# Assign each customer to a segment based on hierarchical clustering
purchase_data['Segment_Hierarchical'] = fcluster(Z, optimal_clusters_hierarchical, criterion='maxclust')

# Display the segment assignments
print("\nSegment Assignments (Hierarchical Clustering):")
purchase_data[['Customer_ID', 'Segment_Hierarchical']].head()


# # Segment Profiling

# In[92]:


# Segment Profiling for K-means Clustering
segment_profiles_kmeans = purchase_data.groupby('Segment_Kmeans').agg({
    'Customer_ID': 'count',
    'Total_Purchase_Amount': 'mean',
    'Average_Purchase_Amount': 'mean'
}).reset_index()

segment_profiles_kmeans.rename(columns={
    'Customer_ID': 'Customer_Count',
    'Total_Purchase_Amount': 'Average_Total_Purchase_Amount',
    'Average_Purchase_Amount': 'Average_Average_Purchase_Amount'
}, inplace=True)

print("Segment Profiles (K-means Clustering):")
print(segment_profiles_kmeans)


# In[93]:


# Segment Profiling for Hierarchical Clustering
segment_profiles_hierarchical = purchase_data.groupby('Segment_Hierarchical').agg({
    'Customer_ID': 'count',
    'Total_Purchase_Amount': 'mean',
    'Average_Purchase_Amount': 'mean'
}).reset_index()

segment_profiles_hierarchical.rename(columns={
    'Customer_ID': 'Customer_Count',
    'Total_Purchase_Amount': 'Average_Total_Purchase_Amount',
    'Average_Purchase_Amount': 'Average_Average_Purchase_Amount'
}, inplace=True)

print("\nSegment Profiles (Hierarchical Clustering):")
print(segment_profiles_hierarchical)


# # Recommendation System Model with Singular Value Decomposition (Turncated)

# In[94]:


# Prepare the historical data on customer-product interactions
X = product_interactions_data.pivot(index='Customer_ID', columns='Product_ID', values='Rating').fillna(0)

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train the recommendation model using TruncatedSVD
svd = TruncatedSVD(n_components=10, random_state=42)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

# Evaluate the model using RMSE (Root Mean Squared Error)
predictions = svd.inverse_transform(X_test_svd)
rmse = mean_squared_error(X_test, predictions, squared=False)
print("RMSE (Root Mean Squared Error):", rmse)


# In[95]:


# Generate recommendations for a sample customer
sample_customer_id = 1509  # Adjust as needed
sample_customer_index = X.index.get_loc(sample_customer_id)
predicted_ratings = svd.inverse_transform(X_train_svd)[sample_customer_index]
top_recommendations = sorted(list(enumerate(predicted_ratings)), key=lambda x: x[1], reverse=True)[:10]
print("\nRecommendations for Customer", sample_customer_id, ":")
for product_index, predicted_rating in top_recommendations:
    product_id = X.columns[product_index]
    print("Product ID:", product_id, "| Predicted Rating:", predicted_rating)


# # Evaluation

# In[96]:


# Prepare data for evaluation
X_eval = product_interactions_data.pivot(index='Customer_ID', columns='Product_ID', values='Rating').fillna(0)

# Initialize evaluation metrics
precision_scores = []
recall_scores = []
average_precision_scores = []

# Perform K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_eval):
    X_train_eval, X_test_eval = X_eval.iloc[train_index], X_eval.iloc[test_index]
    
    # Train the recommendation model using TruncatedSVD
    svd_eval = TruncatedSVD(n_components=10, random_state=42)
    X_train_svd_eval = svd_eval.fit_transform(X_train_eval)
    X_test_svd_eval = svd_eval.transform(X_test_eval)
    
    # Generate recommendations for test set
    predictions_eval = svd_eval.inverse_transform(X_test_svd_eval)
    
    # Flatten the predictions and true ratings
    predictions_flat = predictions_eval.flatten()
    true_ratings_flat = X_test_eval.values.flatten()
    
    # Calculate precision, recall, and average precision
    # Calculate precision, recall, and average precision
    precision = precision_score(true_ratings_flat > 0, predictions_flat > 0, average='binary')
    recall = recall_score(true_ratings_flat > 0, predictions_flat > 0, average='binary')
    
    # Reshape arrays for average_precision_score
    true_ratings_flat = true_ratings_flat.reshape(-1, 1)
    predictions_flat = predictions_flat.reshape(-1, 1)
    average_precision = average_precision_score(true_ratings_flat, predictions_flat)
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    average_precision_scores.append(average_precision)

# Calculate average evaluation metrics
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_avg_precision = sum(average_precision_scores) / len(average_precision_scores)

# Display evaluation results
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average Mean Average Precision:", avg_avg_precision)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




