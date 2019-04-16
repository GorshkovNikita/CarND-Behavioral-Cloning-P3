import csv
from scipy import ndimage
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, Cropping2D

lines = []
root_dir = '../behavioral_training_data/'

with open(root_dir + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
angles = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    image = ndimage.imread(root_dir + 'IMG/' + filename)
    images.append(image)
    angles.append(float(line[3]))

x_train = np.array(images)
y_train = np.array(angles)
image_shape = x_train[0].shape
print(y_train.shape)

input_layer = Input(shape=image_shape)
crop_layer = Cropping2D(((70, 25), (0, 0)))(input_layer)
normalized_layer = Lambda(lambda x: x / 255 + 0.5)(crop_layer)
conv1 = Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu')(normalized_layer)
conv2 = Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu')(conv1)
conv3 = Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu')(conv2)
conv4 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv3)
conv5 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv4)
flatten = Flatten()(conv5)
dense1 = Dense(1164)(flatten)
dense2 = Dense(100)(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(10)(dense3)
output_layer = Dense(1)(dense4)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
