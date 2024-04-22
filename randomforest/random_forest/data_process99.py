
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
from sklearn.ensemble import RandomForestClassifier

cols = """duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,
num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,
is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,
dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,
dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"""
cols = [c.strip() for c in cols.split(",") if c.strip()]
cols.append('target')
# print(cols)
# 标签对应的攻击类型
attacks_type = {
#target : attack
'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}
# print(attacks_type)

df = pd.read_csv('../kdd99/kddcup.data_10_percent', names=cols)
df['Attack'] = df.target.apply(lambda r: attacks_type[r[:-1]])
print("The data shape is (lines, columns):",df.shape)

df_std = df.std(numeric_only=True) #所有特征的方差
df_std = df_std.sort_values(ascending=True) #排序输出

print(df)
def standardize_columns(df):
    zerostdf = df_std[df_std == 0].index
    df = df.drop(zerostdf, axis=1)
    if 'service' in df.columns:
        df = df.drop(['service'], axis = 1)
    return df

df = standardize_columns(df)
df = df.drop(['target',], axis=1)
y = df.Attack
X = df.drop(['Attack',], axis=1)
# print(X)
# print(y)

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

y_train = le_y.fit_transform(y_train.values)
y_test = le_y.transform(y_test.values)

#保存标签
joblib.dump(le_X_cols, 'le_X_cols.pkl') 
joblib.dump(le_y, 'le_y.pkl') 

class_names, class_index = le_y.classes_, np.unique(y_train)
print(class_names, class_index)

#特征缩放
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train[['dst_bytes','src_bytes']] = scaler.fit_transform(X_train[['dst_bytes','src_bytes']])
X_test[['dst_bytes','src_bytes']] = scaler.transform(X_test[['dst_bytes','src_bytes']])
#保存
joblib.dump(scaler, 'scaler_1.pkl')


#随机森林
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 30)
rf.fit(X_train, y_train)
print("=========================================================")
print("随机森林训练准确度:", rf.score(X_train, y_train))
print("随机森林测试准确度:",rf.score(X_test,y_test))
diff_base = abs(rf.score(X_train, y_train) - rf.score(X_test,y_test))
print("模型的过度/不足拟合：", diff_base)
#预测测试集中的数据
rf_pred = rf.predict(X_test)
reversefactor = dict(zip(class_index,class_names))
y_test_rev = np.vectorize(reversefactor.get)(y_test)
y_pred_rev = np.vectorize(reversefactor.get)(rf_pred)
#生成混淆矩阵
print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))
print("=========================================================")

#决策树
from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(X_train, y_train)
print("=========================================================")
print("决策树训练准确度:", dc.score(X_train, y_train))
print("决策树测试准确度:", dc.score(X_test,y_test))
diff_base = abs(dc.score(X_train, y_train) - dc.score(X_test,y_test))
print("模型的过度/不足拟合：", diff_base)
dc_pred = dc.predict(X_test)
y_pred_rev = np.vectorize(reversefactor.get)(dc_pred)
print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))
print("=========================================================")

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print("=========================================================")
print("knn训练准确度:", knn.score(X_train, y_train))
print("knn测试准确度:", knn.score(X_test,y_test))
diff_base = abs(dc.score(X_train, y_train) - dc.score(X_test,y_test))
print("模型的过度/不足拟合：", diff_base)
knn_pred = knn.predict(X_test)
y_pred_rev = np.vectorize(reversefactor.get)(knn_pred)
print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))
print("=========================================================")
