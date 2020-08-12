#%%

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as py
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K

f = Dataset(r"C:\Users\rampaln\Downloads\Training_Dataset_Final.nc")
training_data = f['Reflectance'][:].data
X_train = training_data[0:1000]
X_test = training_data[1000:]



fig, ax = py.subplots(4,4, figsize = (8,8))
ax = ax.ravel()
for i, axes in enumerate(ax):
    axes.imshow(X_train[i], cmap ='gray')
    axes.set_yticklabels([])
    axes.set_xticklabels([])


input_img = Input(shape=(50, 50, 1))  # adapt this if using `channels_first` image data format
x = Conv2D(16, (9, 9), activation='relu', padding='same')(input_img)
x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
#x 
#x = Conv2D(512, (7, 7), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (2, 2), activation='relu', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')



autoencoder.summary()



sub_sample_train = X_train[:,::4,::4,np.newaxis]
sub_sample_test = X_test[:,::4,::4,np.newaxis]
autoencoder.fit(sub_sample_train, X_train[:,:,:,np.newaxis],
                epochs=10,
                batch_size=40,
                shuffle=True,
                validation_data=(sub_sample_test, X_test[:,:,:,np.newaxis]))#,

img = autoencoder.predict(X_train[:,::4,::4,np.newaxis])
autoencoder.save('model.h5')

idx_of_interest = 8
fig, ax = py.subplots(1,3, figsize = (20,20))
ax[0].imshow(X_train[:,::4,::4,np.newaxis][idx_of_interest,:,:,0], cmap ='gray')
ax[1].imshow(img[idx_of_interest,:,:,0], cmap='gray')
ax[2].imshow(X_train[idx_of_interest],cmap='gray')
fig.savefig('test_image.pdf', dpi =300)

