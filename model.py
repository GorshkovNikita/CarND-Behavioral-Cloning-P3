import csv
from scipy import ndimage
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
root_dir = '../data/'

with open(root_dir + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# generator, which allows not to store all training data in memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                center_image_filename = 'IMG/' + batch_sample[0].split('/')[-1]
                left_image_filename = 'IMG/' + batch_sample[1].split('/')[-1]
                right_image_filename = 'IMG/' + batch_sample[2].split('/')[-1]

                center_image = ndimage.imread(root_dir + center_image_filename)
                left_image = ndimage.imread(root_dir + left_image_filename)
                right_image = ndimage.imread(root_dir + right_image_filename)

                center_image_flipped = np.fliplr(center_image)
                left_image_flipped = np.fliplr(left_image)
                right_image_flipped = np.fliplr(right_image)
                correction = 0.1
                center_angle = float(batch_sample[3])
                right_angle = center_angle - correction
                left_angle = center_angle + correction
                images.extend([
                    center_image, left_image, right_image,
                    center_image_flipped, left_image_flipped, right_image_flipped
                ])
                steering_angles.extend([
                    center_angle, left_angle, right_angle,
                    -center_angle, -left_angle, -right_angle
                ])

            x_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(x_train, y_train)


batch_size=256

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

sample_image = ndimage.imread(root_dir + 'IMG/' + lines[0][0].split('/')[-1])
print(sample_image.shape)

# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
input_layer = Input(shape=sample_image.shape)
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
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=np.ceil(len(validation_samples) / batch_size),
    epochs=5
)

model.save('model.h5')
