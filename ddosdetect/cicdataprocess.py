import pandas as pd
import numpy as np
import os
import sys
path1 = './dataset/cicddos2019/csv-03-11/03-11/' #for train
path2 = './dataset/cicddos2019/csv-01-12/01-12/' #for test
filelist1 = os.listdir(path1)
filelist2 = os.listdir(path2)

filelist1 = [f for f in filelist1 if os.path.isfile(os.path.join(path1, f))]
filelist2 = [f for f in filelist2 if os.path.isfile(os.path.join(path2, f))]

print(filelist1)
print(filelist2)
# sys.exit()

drop_columns = ["Unnamed: 0", "Source Port", "Flow ID", "Source IP", "Destination IP",
                "SimillarHTTP", "Flow Bytes/s", "Flow Packets/s"]



# def grep_benigns(path, filename):
#     benigns = pd.DataFrame()
#     for chunk in pd.read_csv(path+filename, chunksize = 150000, low_memory = False):
#         benigns = pd.concat([benigns, chunk.drop(chunk.loc[chunk[' Label'] != 'BENIGN'].index)])
#     benigns.columns = benigns.columns.str.strip()
#     benigns.to_csv(path + 'benign/' + 'BENIGN_' + filename, index=False)

# def clean_benigns(dataframe, path):
#     dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
#     dataframe.dropna(inplace = True)
#     dataframe.drop(columns = drop_columns, inplace=True)
#     dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
#     dataframe.sort_values(by='Timestamp', inplace=True)
#     dataframe.drop(columns = 'Timestamp', inplace=True)
#     dataframe.to_csv(path + 'benign/' + 'BENIGN_CLEANED_ALL_' + '.csv', index=False)

# # for file in filelist1:
# #     grep_benigns(path1, file)

# # for file in filelist2:
# #     grep_benigns(path2, file)

# pd.concat([pd.read_csv(path1 + 'benign/' + 'BENIGN_' + f) for f in filelist1]).to_csv(
#     path1 + 'benign/' + 'BENIGN_all_'  + '.csv', index=False)

# pd.concat([pd.read_csv(path2 + 'benign/' + 'BENIGN_' + f) for f in filelist2]).to_csv(
#     path2 + 'benign/' + 'BENIGN_all_'  + '.csv', index=False)

# df_benign1 = pd.read_csv(path1 + 'benign/BENIGN_all_.csv')
# df_benign2 = pd.read_csv(path2 + 'benign/BENIGN_all_.csv')

# clean_benigns(df_benign1, path1)
# clean_benigns(df_benign2, path2)

frac = 0.01
fracsize = str(frac * 100)


def chunk_sampling_x(path, filename, fracsize):
    frac = str(fracsize * 100)
    samples = pd.DataFrame()
    for chunk in pd.read_csv(path+filename, chunksize = 150000, low_memory = False):
        samples = pd.concat([samples, chunk.sample(frac=fracsize, random_state=62)])
    samples.columns = samples.columns.str.strip()
    samples.to_csv(path + "fracsample/" + "strat_" + frac + "_" + filename, index=False)
    
def clean(path, filename, drop_columns):
    _path = path + "fracsample/"
    original = pd.read_csv(_path+filename, low_memory = False)
    original.replace([np.inf, -np.inf], np.nan, inplace=True)
    original.dropna(inplace = True)
    original.drop(columns = drop_columns, inplace=True)
    original.to_csv(path + "clean/" + "CLEAN_" + filename, header=True, index=False)
    del original

def sort_drop_timestamp(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.sort_values(by='Timestamp', ignore_index=True)
    df.drop(columns = 'Timestamp', inplace=True)

# for file in filelist1:
#     chunk_sampling_x(path1, file, frac)

# for file in filelist2:
#     chunk_sampling_x(path2, file, frac)

# for file in filelist1:
#     file = 'strat_' + fracsize + '_' + file
#     clean(path1, file, drop_columns)

# for file in filelist2:
#     file = 'strat_' + fracsize + '_' + file
#     clean(path2, file, drop_columns)

# pd.concat([pd.read_csv(path1 + "clean/" +"CLEAN_strat_" + fracsize + "_" + f) for f in filelist1]).to_csv(
#     path1 + "clean/" + "cleaned_all_" + fracsize + "_" + ".csv", index=False)

# pd.concat([pd.read_csv(path2 + "clean/" +"CLEAN_strat_" + fracsize + "_" + f) for f in filelist2]).to_csv(
#     path2 + "clean/" + "cleaned_all_" + fracsize + "_" + ".csv", index=False)


cic_1 = pd.read_csv(path1 + 'clean/cleaned_all_' + fracsize + '_.csv', low_memory=False)
cic_2 = pd.read_csv(path2 + 'clean/cleaned_all_' + fracsize + '_.csv', low_memory=False)

sort_drop_timestamp(cic_1)
sort_drop_timestamp(cic_2)

cic_2['Label'].replace({'UDP-lag':'UDPLag', 'DrDoS_UDP':'UDP', 'DrDoS_LDAP':'LDAP', 'DrDoS_MSSQL':'MSSQL', 'DrDoS_NetBIOS':'NetBIOS',
                           'DrDoS_DNS':'DNS', 'DrDoS_NTP':'NTP', 'DrDoS_SSDP':'SSDP', 'DrDoS_SNMP':'SNMP'}, inplace = True)

cic_1.drop(cic_1.loc[cic_1['Label'] == 'BENIGN'].index, inplace=True)
cic_2.drop(cic_2.loc[cic_2['Label'] == 'BENIGN'].index, inplace=True)

df_benign_clean1 = pd.read_csv(path1 + 'benign/BENIGN_CLEANED_ALL_.csv')
df_benign_clean2 = pd.read_csv(path2 + 'benign/BENIGN_CLEANED_ALL_.csv')

cic_1 = pd.concat([cic_1, df_benign_clean1], ignore_index=True)
cic_2 = pd.concat([cic_2, df_benign_clean2], ignore_index=True)

# cic_1 = cic_1.sample(frac=1).reset_index(drop=True)
# cic_2 = cic_2.sample(frac=1).reset_index(drop=True)

cic_1.to_csv(path1 + fracsize + '_percent_1' + '.csv', index=False)
cic_2.to_csv(path2 + fracsize + '_percent_1' + '.csv', index=False)

del cic_1
del cic_2

# df_test_cic_nineteen = pd.read_csv(day1_path + fracsize + '_percent_' + day1_path[-5:-1] + '.csv', low_memory=False)