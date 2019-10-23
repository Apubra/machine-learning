from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
print('---------------')
print(scaler.data_max_)
print('---------------')
print(scaler.data_min_)
print('---------------')

print(scaler.transform(data))

print('---------------')

print(scaler.transform([[2, 2]]))
print('-------FINAL--------')
import numpy as np
dat = np.array([[5000], [700]])
print(scaler.fit_transform(dat))