from data_reader import readFile
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

fsData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/breast_swimming1.txt'
fdData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/breast_drowning1.txt'
fdData2 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/crawl_drowning1.txt'
fsData2 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/crawl_swimming1.txt'


files = [fdData2, fsData1, fsData2]

read1 = readFile(fdData1)
df1 = pd.DataFrame(read1)
df1 = df1.drop('tfps', axis=1)
df1['activity'] = 0
read2 = readFile(fdData2)
df2 = pd.DataFrame(read2)
df2 = df2.drop('tfps', axis=1)
df2['activity'] = 0
read3 = readFile(fsData1)
df3 = pd.DataFrame(read3)
df3 = df3.drop('tfps', axis=1)
df3['activity'] = 1
read4 = readFile(fsData2)
df4 = pd.DataFrame(read4)
df4 = df4.drop('tfps', axis=1)
df4['activity'] = 2

data = pd.concat([df1, df2, df3, df4])

data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')
print(data['activity'].value_counts())
df = data
drowning = df[df['activity']==0].head(3100)
breast = df[df['activity']==1].head(3100)
crawl = df[df['activity']==2].head(3100)


balanced_data = pd.DataFrame()
balanced_data = pd.concat([drowning, breast, crawl])
print(balanced_data.shape)

label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
balanced_data.head()

X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']
print(label.classes_)

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values

print(scaled_X.head())

df.to_csv('scaled_data00.csv')

