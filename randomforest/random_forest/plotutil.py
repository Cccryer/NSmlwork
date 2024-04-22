import matplotlib.pyplot as plt 
import numpy as np 

# 分布直方图
import numpy as np
import matplotlib.pyplot as plt

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 70]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow
    
    # 调整图形大小和分辨率
    plt.figure(figsize=(40, 40))
    
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot(kind='bar', color='skyblue', edgecolor='black')
        else:
            columnDf.hist(color='lightcoral', edgecolor='black')
        plt.ylabel('Counts', fontsize=15)
        plt.xlabel(columnNames[i], fontsize=15)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(f'{columnNames[i]}', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        
    # 改善布局
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()
    
    # 将保存图形的代码移到显示图形之前
    plt.savefig('./datadistribute.png', dpi=150)



# 相关矩阵
def plotCorrelationMatrix(df, graphWidth, dataframeName):
    filename = dataframeName#df.dataframeName
    df = df.dropna() #舍去值为NaN的列
    df = df[[col for col in df if df[col].nunique() > 1]] #保留拥有多于一个唯一值的列
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr(numeric_only=True)
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    plt.savefig('./correlationMatrix')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    # 计算混淆矩阵
    class_names = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('KNN Confusion Matrix')
    plt.show()
    plt.savefig('./confusion')

