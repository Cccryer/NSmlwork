
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
from sklearn.ensemble import RandomForestClassifier
from plotutil import plotPerColumnDistribution, plotCorrelationMatrix, plot_confusion_matrix
import sys
import time
#数据记录的42项特征
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
value_counts = df['Attack'].value_counts()
print(value_counts)
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar')
plt.title('Value Counts of ' + 'Attack Type')
plt.xlabel('Attack Type')
plt.ylabel('Counts')
plt.show()
plt.savefig('./attackcount')


print(df.Attack.value_counts())
print(df.target.unique())
print(df.Attack.unique())
plotPerColumnDistribution(df[[
    'protocol_type',
    'service',
    'flag',
    #'logged_in',
    # 'srv_serror_rate',
    # 'srv_diff_host_rate',
    'Attack'
]], nGraphShown=30, nGraphPerRow=2)
# sys.exit()
plotCorrelationMatrix(df, graphWidth=20, dataframeName="Packets")
# sys.exit()
for c in df.columns:
    print("%30s : %d"%(c, sum(pd.isnull(df[c]))))
df_std = df.std(numeric_only=True) 
df_std = df_std.sort_values(ascending=True) 
print(df_std)
plt.figure(figsize=(15,10))
plt.plot(list(df_std.index) ,list(df_std.values), 'go')

plt.show()
plt.savefig("./test")

#提前通过plotScatterMatrix观察过相关矩阵
print(df)
def standardize_columns(df):
    #删除'service'列；如果存在TCPDUMP列则重命名
    zerostdf = df_std[df_std == 0].index
    df = df.drop(zerostdf, axis=1)
    if 'service' in df.columns:
        df = df.drop(['service'], axis = 1)
    #df.rename(columns = cols_map)
    return df

df = standardize_columns(df)
df = df.drop(['target',], axis=1)
y = df.Attack
X = df.drop(['Attack',], axis=1)
# print(X)
# print(y)
#随机生成训练集、测试集
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


scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train[['dst_bytes','src_bytes']] = scaler.fit_transform(X_train[['dst_bytes','src_bytes']])
X_test[['dst_bytes','src_bytes']] = scaler.transform(X_test[['dst_bytes','src_bytes']])
#保存
joblib.dump(scaler, 'scaler_1.pkl')


# #随机森林
# rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
# time_start=time.time()
# rf.fit(X_train, y_train)
# print("=========================================================")
# trainacc = rf.score(X_train, y_train)
# testacc = rf.score(X_test,y_test)
# time_end=time.time()
# print('time cost',time_end-time_start,'s')
# print("随机森林训练准确度:", trainacc)
# print("随机森林测试准确度:",testacc)
# diff = abs(trainacc - testacc)
# print("模型的过度/不足拟合：", diff)
# #预测测试集中的数据
# rf_pred = rf.predict(X_test)
# reversefactor = dict(zip(class_index,class_names))
# y_test_rev = np.vectorize(reversefactor.get)(y_test)
# y_pred_rev = np.vectorize(reversefactor.get)(rf_pred)

# #生成混淆矩阵
# plot_confusion_matrix(y_test_rev, y_pred_rev)
# print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))
# print("=========================================================")

# sys.exit()

# #决策树
# from sklearn.tree import DecisionTreeClassifier
# dc = DecisionTreeClassifier()
# time_start=time.time()
# dc.fit(X_train, y_train)
# print("=========================================================")
# trainacc = dc.score(X_train, y_train)
# testacc = dc.score(X_test,y_test)
# time_end=time.time()
# print('time cost',time_end-time_start,'s')
# print("决策树训练准确度:", trainacc)
# print("决策树测试准确度:", testacc)
# diff = abs(trainacc - testacc)
# print("模型的过度/不足拟合：", diff)
# dc_pred = dc.predict(X_test)
# reversefactor = dict(zip(class_index,class_names))
# y_test_rev = np.vectorize(reversefactor.get)(y_test)
# y_pred_rev = np.vectorize(reversefactor.get)(dc_pred)

# plot_confusion_matrix(y_test_rev, y_pred_rev)
# print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))
# print("=========================================================")


# sys.exit()

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
time_start=time.time()
knn.fit(X_train, y_train)
print("=========================================================")
trainacc = knn.score(X_train, y_train)
testacc = knn.score(X_test,y_test)
time_end=time.time()
print('time cost',time_end-time_start,'s')
print("knn训练准确度:", trainacc)
print("knn测试准确度:", testacc)
diff = abs(trainacc - testacc)
print("模型的过度/不足拟合：", diff)
knn_pred = knn.predict(X_test)
reversefactor = dict(zip(class_index,class_names))
y_test_rev = np.vectorize(reversefactor.get)(y_test)
y_pred_rev = np.vectorize(reversefactor.get)(knn_pred)
print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))
plot_confusion_matrix(y_test_rev, y_pred_rev)
print("=========================================================")




# clf = RandomForestClassifier(n_estimators=30)
# clf = clf.fit(X_train, y_train)
# fti = clf.feature_importances_
# model = SelectFromModel(clf, prefit=True, threshold= 0.005)
# X_train_new = model.transform(X_train.values)
# X_test_new = model.transform(X_test.values)
# selcted_features = X_train.columns[model.get_support()]
# print(X_train_new.shape)

# print(selcted_features)


# parameters = {
#     'n_estimators'      : [20,40,128,130],
#     'max_depth'         : [None,14, 15, 17],
#     'criterion' :['gini','entropy'],
#     'random_state'      : [42],
#     #'max_features': ['auto'],
    
# }
# clf = GridSearchCV(RandomForestClassifier(), parameters, cv=2, n_jobs=-1, verbose=5)
# clf.fit(X_train_new, y_train)

# print("clf.best_estimator_:",clf.best_estimator_)
# print("clf.best_params_",clf.best_params_)
# #print("results:",clf.cv_results_)

# print("CV训练准确率:",clf.best_score_)
# print("CV测试准确率:",clf.score(X_test_new,y_test))
# diff_fst = abs(clf.best_score_ - clf.score(X_test_new,y_test))
# print("准确率差：", diff_fst)
# print("模型表现提升？", diff_base > diff_fst)

# #混淆矩阵
# #预测测试数据集
# y_pred = clf.predict(X_test_new)

# reversefactor = dict(zip(class_index,class_names))
# y_test_rev = np.vectorize(reversefactor.get)(y_test)
# y_pred_rev = np.vectorize(reversefactor.get)(y_pred)
# #生成混淆矩阵
# print(pd.crosstab(y_test_rev, y_pred_rev, rownames=['Actual packets attacks'], colnames=['Predicted packets attcks']))

# #fig, ax = plt.subplots(figsize=(15, 10))
# #plot.confusion_matrix(y_test_rev, y_pred_rev, ax=ax)
# #plt.show()

# joblib.dump(clf, 'random_forest_classifier.pkl') 
# #To load it: clf_load = joblib.load('saved_model.pkl') 