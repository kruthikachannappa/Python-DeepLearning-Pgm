import numpy as np
x=np.random.randint(1,20,15)
print("Original array:")
print(x)
x[x.argmax()] = 0
print("Maximum value replaced by 0:")
print(x)