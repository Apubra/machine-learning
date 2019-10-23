import pandas as pd
iris = pd.read_csv('/media/apubra/e3371ede-8799-41bd-8f5f-9ef53c1ee7fa/Git Repository/machine-learning--basic/datasets/iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(iris.head())

import matplotlib.pyplot as plt
# # create a figure and axis
# fig, ax = plt.subplots()

# # scatter the sepal_length against the sepal_width
# ax.scatter(iris['sepal_length'], iris['sepal_width'], iris['petal_length'])
# # set a title and labels
# ax.set_title('Iris Dataset')
# ax.set_xlabel('sepal_length')
# ax.set_ylabel('sepal_width')

# create color dictionary
colors = ['#2300A8', '#00A658']
# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
ax.scatter(iris['sepal_length'], iris['sepal_length'],color=['red'],marker='*')
ax.scatter(iris['petal_length'], iris['petal_length'],color=['green'])
# set a title and labels
ax.set_title('Iris Dataset')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
plt.show()