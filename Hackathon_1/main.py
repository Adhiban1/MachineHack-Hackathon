import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
# print(train, test, sep='\n')

x_cols = ['trip_duration', 'distance_traveled', 'num_of_passengers', 'surge_applied']
y_col = ['total_fare']
x_train, y_train, x_test, y_test = train[x_cols], train[y_col], test[x_cols], test[y_col]

for i in [x_train, y_train, x_test, y_test]:
    print(i.shape)

model = RandomForestRegressor(n_estimators=10)
model.fit(x_train, y_train)
predicted = model.predict(x_test)
score = np.sqrt(mean_squared_log_error(y_test, predicted))
print(score)
