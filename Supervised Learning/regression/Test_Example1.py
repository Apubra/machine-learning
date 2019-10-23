from sklearn.linear_model import LinearRegression
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = LinearRegression()
reg.fit(X, Y)  
res = reg.predict([[1, 0.]])
print(res)