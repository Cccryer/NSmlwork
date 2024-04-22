import pandas as pd
from sklearn.preprocessing import LabelEncoder
# pd.set_option('display.max_columns', None)

# df = pd.read_csv('./alldata1.csv', nrows =10)
# print(df)

labeldf = pd.read_csv('./alldata1.csv', usecols= [' Label'])


label_encoder = LabelEncoder()
labeldf[' Label'] = label_encoder.fit_transform(labeldf[' Label'])
    # Get the mapping between original labels and encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # Print the mapping
for label, encoded_value in label_mapping.items():
    print(f"Label: {label} - Encoded Value: {encoded_value}")


labeldf.to_csv('./alldata1.csv', index=False, columns=[' Label'])