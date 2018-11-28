from sklearn.datasets import fetch_mldata

import matplotlib.pyplot as plt

# -------------------------------------------

mnist = fetch_mldata('MNIST original')

print(mnist)
print(mnist.data)
print(mnist.target)
len(mnist.data)
help(len)
print(mnist.data.shape)
print(mnist.target.shape)
mnist.data[0]
mnist.data[0][1]
mnist.data[:,1]
mnist.data[:100]

# -------------------------------------------

mnist = fetch_mldata('MNIST original')
images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
