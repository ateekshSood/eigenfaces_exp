# %%

import numpy as np 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# %%

x , y  = fetch_olivetti_faces(random_state= 42 , return_X_y= True)

x_train , _ , y_train , _ = train_test_split(x , y  , random_state = 42)

# %%

print(x_train.shape)
import matplotlib.pyplot as plt 

plt.imshow(x_train[ 0 , :].reshape(np.sqrt(x_train.shape[1]).astype(int) ,np.sqrt(x_train.shape[1]).astype(int) , -1) , cmap='binary')
plt.axis("off")
plt.show()

# %%
    