from zipfile import ZipFile
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def leaderboard():
    with open('text.txt') as f:
        data = f.read()
    data = re.findall('[\d]+\.([\d]+.[\d]+)', data)
    return [float(i[5:]) for i in data]

if not os.path.exists('dataset'):
    with ZipFile('Participants_Data_GGSH_Solution_Notebook.zip') as f:
        f.extractall('dataset')

train = pd.read_csv('dataset/India_train.csv')
test = pd.read_csv('dataset/India_test.csv')
submission = pd.read_csv('dataset/submission.csv')

print('Train columns: ')
for i,j in enumerate(train.columns):
    print(f"{i+1:>3}. {j} | {train[j].dtype}")

print('\nTo Predict columns: ')
for i, j in enumerate(submission.columns):
    print(f"{i+1}. {j}", end='')
    if j in train.columns:
        print(' ✅')
        y_column = j
    else:
        print(' ❌')

print(f'\n{any(train.columns == test.columns) = }')

train1 = train.select_dtypes(['int64', 'float64'])
test1 = test.select_dtypes(['int64', 'float64'])

for i in train1.columns:
    train1[i] = train1[i].fillna(train1[i].mean())
    test1[i] = test1[i].fillna(test1[i].mean())

x_train, y_train, x_test, y_test = (train1.drop(y_column, axis=1), train1[y_column],
                                    test1.drop(y_column, axis=1), test1[y_column])

x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

model = AdaBoostRegressor(n_estimators=10)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f'\nScore: {score}')

mse = mean_squared_error(y_test, model.predict(x_test))
print(f'MSE: {mse} | {(mse < 13147049.32475) = }')
