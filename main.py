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

#getting the mean of the faces 

x_train_mean = np.mean(x_train , axis=0)

x_train_centered = x_train - x_train_mean


x_train_mean.shape


plt.imshow(x_train_mean.reshape(np.sqrt(x_train_mean.shape[0]).astype(int) , np.sqrt(x_train_mean.shape[0]).astype(int) , -1) , cmap='binary')
plt.axis('off')
plt.show()
# %%
fig , ax = plt.subplots(1 , 2 )

ax[0].imshow(x_train[ 0 , :].reshape(np.sqrt(x_train.shape[1]).astype(int) ,np.sqrt(x_train.shape[1]).astype(int) , -1) , cmap='binary')
ax[0].axis('off')
ax[0].set_title("Normal")

ax[1].imshow(x_train[ 0 , :].reshape(np.sqrt(x_train_centered.shape[1]).astype(int) ,np.sqrt(x_train_centered.shape[1]).astype(int) , -1) , cmap='binary')
ax[1].axis('off')
ax[1].set_title("Centered")

plt.show()

# %%

covariance_matrix = 1/(x_train_centered.shape[0]) * (x_train_centered.T @ x_train_centered)
eigenvalues , eigenvectors = np.linalg.eigh(covariance_matrix)
# %%

sorted_eigenvalues_indexes = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_eigenvalues_indexes]
sorted_eigenvectors = eigenvectors[: , sorted_eigenvalues_indexes]

# %%

print(eigenvectors.shape)


eigenvalues.shape
# %%

# cartesian product
# 
from itertools import product

fig , ax = plt.subplots(3 , 3)

for w,(i,z) in enumerate(product(range(3) , range(3))):

    ax[i , z].imshow(sorted_eigenvectors[: , w].reshape(64 , 64 , -1) , cmap="binary")
    ax[i, z].axis('off')

# %%


#findiunig the k 


k = np.argmax(np.cumsum(sorted_eigenvalues) / total_eigenvalues_sum > 0.95)
print(k)
# %%

test_face = x_train[0]

used_principal_components = sorted_eigenvectors[: , :k]


compress = test_face @ used_principal_components
compress.shape

decompress = compress @ used_principal_components.T

final_reconstructed_face = decompress + x_train_mean

plt.imshow(final_reconstructed_face.reshape(64 , 64 , -1) , cmap='binary')
plt.axis('off')
plt.show()

# %%
plt.imshow(x_train[0].reshape(64 , 64 , -1) , cmap="binary")
plt.axis('off')
plt.show()

