# Iris Flower Classification
# Import Packages
import boto3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle5 as pickle
#%matplotlib inline

sts_client = boto3.client('sts')
assumed_role = sts_client.assume_role(
    RoleArn='arn:aws:iam::430238166084:role/s3_role',  # Replace with your role ARN
    RoleSessionName='mltest'  # A name for the session
)
"""
credentials = assumed_role['Credentials']   
s3_client = boto3.client(
    's3',
    #aws_access_key_id=credentials['AccessKeyId'],
    #aws_secret_access_key=credentials['SecretAccessKey'],
    aws_access_key_id="AKIAWILBRNRCO2LSFMWH",
    aws_secret_access_key="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"
)

bucket_name = 'ml-iris-chetan' 
#s3_client.download_file(bucket_name, 'iris.data.csv', '/home/chetan/Desktop/gitrepo/iris_ml/Iris_ML/iris.data.csv')
#s3_client.download_file(bucket_name, 'iris.data.csv', '/home/chetan/Desktop/gitrepo/iris_ml/Iris_ML/iris_data.csv')
"""
"""
result = s3_client.list_buckets()
for i in result['Buckets']:
    print (i['Name'])
"""

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv('iris_data.csv', names=columns)
print (df.info)
print (df.describe())
print (sns.pairplot(df, hue='Class_labels'))

#plt.show()

# Separate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]

"""
# Calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1])
 for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25

# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
#plt.show()
"""

# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)
# Calculate the accuracy

print (accuracy_score(y_test, predictions))

# create an iterator object with write permission - model.pkl
with open('model_pkl', 'wb') as files:
    pickle.dump(svn, files)
