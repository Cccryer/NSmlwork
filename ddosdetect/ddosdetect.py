import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
from sklearn import preprocessing
from datetime import datetime

data = pd.read_csv('./dataset/dataset_sdn.csv')
print(data.head)
print(data.shape)
print(data.info())
print(data.label.unique())
print(data.label.value_counts())
print(data.isnull().sum())


numeric_df = data.select_dtypes(include=['int64', 'float64'])
object_df = data.select_dtypes(include=['object'])
numeric_cols = numeric_df.columns
object_cols = object_df.columns
print('Numeric Columns: ')
print(numeric_cols, '\n')
print('Object Columns: ')
print(object_cols, '\n')
print('Number of Numeric Features: ', len(numeric_cols))
print('Number of Object Features: ', len(object_cols))
print(object_df.head())

figure(figsize=(12, 7), dpi=80)
plt.barh(list(dict(data.src.value_counts()).keys()), dict(data.src.value_counts()).values(), color='lawngreen')
plt.barh(list(dict(data[data.label == 1].src.value_counts()).keys()), dict(data[data.label == 1].src.value_counts()).values(), color='blue')

for idx, val in enumerate(dict(data.src.value_counts()).values()):
    plt.text(x = val, y = idx-0.2, s = str(val), color='r', size = 13)

for idx, val in enumerate(dict(data[data.label == 1].src.value_counts()).values()):
    plt.text(x = val, y = idx-0.2, s = str(val), color='w', size = 13)


plt.xlabel('Number of Requests')
plt.ylabel('IP addres of sender')
plt.legend(['All','malicious'])
plt.title('Number of requests from different IP adress')
plt.savefig('./Number of requests from different IP adress')


figure(figsize=(10, 6), dpi=80)
plt.bar(list(dict(data.Protocol.value_counts()).keys()), dict(data.Protocol.value_counts()).values(), color='r')
plt.bar(list(dict(data[data.label == 1].Protocol.value_counts()).keys()), dict(data[data.label == 1].Protocol.value_counts()).values(), color='b')

plt.text(x = 0 - 0.15, y = 41321 + 200, s = str(41321), color='black', size=17)
plt.text(x = 1 - 0.15, y = 33588 + 200, s = str(33588), color='black', size=17)
plt.text(x = 2 - 0.15, y = 29436 + 200, s = str(29436), color='black', size=17)

plt.text(x = 0 - 0.15, y = 9419 + 200, s = str(9419), color='w', size=17)
plt.text(x = 1 - 0.15, y = 17499 + 200, s = str(17499), color='w', size=17)
plt.text(x = 2 - 0.15, y = 13866 + 200, s = str(13866), color='w', size=17)

plt.xlabel('Protocol')
plt.ylabel('Count')
plt.legend(['All', 'malicious'])
plt.title('The number of requests from different protocols')
plt.savefig('./number of requests from different protocols')

df_std = data.std(numeric_only=True) #所有特征的方差
df_std = df_std.sort_values(ascending=True) #排序输出
print(df_std)

X = data.drop('label', axis=1)
y = data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

le_X_cols = {}
le_y = preprocessing.LabelEncoder()

for c in X_train.columns:
    if str(X_train[c].dtype) == 'object': 
        le_X = preprocessing.LabelEncoder()
        X_train[c] = le_X.fit_transform(X_train[c])
        X_test[c] = le_X.transform(X_test[c])
        le_X_cols[c] = le_X


joblib.dump(le_X_cols, 'le_X_cols.pkl') 

print(X_train[:10])

data['dt'] = data['dt'].astype(str)
data['dt'] = data['dt'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))
print(data[:10])

