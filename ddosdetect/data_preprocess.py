import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)

def preprocess(df):
    df = df.replace('Infinity','0')
    df = df.replace(np.inf,0)
    #df = df.replace('nan','0')
    df[' Flow Packets/s'] = pd.to_numeric(df[' Flow Packets/s'])

    df['Flow Bytes/s'] = df['Flow Bytes/s'].fillna(0)
    df['Flow Bytes/s'] = pd.to_numeric(df['Flow Bytes/s'])
        
    #Timestamp - Drop day, then convert hour, minute and seconds to hashing 
    colunaTime = pd.DataFrame(df[' Timestamp'].str.split(' ').tolist(), columns = ['day','hour'])
    colunaTime = pd.DataFrame(colunaTime['hour'].str.split('.').tolist(),columns = ['hour','milisec'])
    stringHoras = pd.DataFrame(colunaTime['hour'].str.encode('utf-8'))
    df[' Timestamp'] = pd.DataFrame(stringHoras['hour'].apply(string2numeric_hash))#colunaTime['horas']
    del colunaTime,stringHoras

    return df



dir = './dataset/cicddos2019/csv-03-11/03-11'

filelist = os.listdir(dir)
print(filelist)
dflist = []
chunk_size = 100000
cols = pd.read_csv(os.path.join(dir, filelist[0]), nrows = 1)
notusecols = [' Source IP', ' Destination IP', 'Flow ID', 'SimillarHTTP', 'Unnamed: 0']
usecolss = [i for i in cols if i not in notusecols]
for filename in filelist:
    filepath = os.path.join(dir, filename)
    chunks = pd.read_csv(filepath, chunksize=chunk_size, usecols=usecolss)
    for chunk in chunks:
        df = preprocess(chunk)
        df.to_csv('./alldata.csv', index = False, mode = 'a')
        #dflist.append(df)


#在读取alldata.csv label列进行编码
labeldf = pd.read_csv('./alldata.csv', usecols= [' Label'])


label_encoder = LabelEncoder()
labeldf[' Label'] = label_encoder.fit_transform(labeldf[' Label'])
    # Get the mapping between original labels and encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # Print the mapping
for label, encoded_value in label_mapping.items():
    print(f"Label: {label} - Encoded Value: {encoded_value}")


labeldf.to_csv('./alldata.csv', index=False, columns=[' Label'])


