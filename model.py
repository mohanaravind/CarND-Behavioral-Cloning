import csv
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D, BatchNormalization
from sklearn.model_selection import  train_test_split
from sklearn.utils import  shuffle



class HyperParameters:
    def __init__(self, epochs=2, batch_size=64, scale=4, correction=0.4, crop_area=((70, 25), (0, 0))):
        self.epochs = epochs
        self.batch_size = batch_size
        self.scale = scale
        self.correction = correction
        self.crop_area = crop_area

        # Other parameters
        self.num_samples = None
        self.input_shape = None

def extract_samples():
    samples = []
    with open('data/driving_log.csv') as log_file:
        # Skip the first line (CSV header)
        next(log_file, None)
        reader = csv.reader(log_file)

        # Read all the lines from csv file
        for line in reader:
            samples.append(line)

    return train_test_split(samples, test_size=0.2)

def read_image(path):
    # Remove white spaces
    path = path.strip()

    # If its a relative path
    if path.startswith('IMG'):
        path = 'data/{}'.format(path)

    return np.array(cv2.imread(path))

def batch_generator(samples, batch_size, correction):
    num_samples = len(samples)

    while 1:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_samples = samples[offset:end]

            images = []
            angles = []

            for batch_sample in batch_samples:
                steering_angle = float(batch_sample[3])

                # Center
                center_image = read_image(batch_sample[0])
                center_image_flp = np.fliplr(center_image)
                center_steer = steering_angle
                center_steer_flp = - center_steer

                # Left
                left_image = read_image(batch_sample[1])
                left_image_flp = np.fliplr(left_image)
                left_steer = steering_angle + correction
                left_steer_flp = -left_steer

                # Right
                right_image = read_image(batch_sample[2])
                right_image_flp = np.fliplr(right_image)
                right_steer = steering_angle - correction
                right_steer_flp = -right_steer

                # Add all the data to the batch
                images.extend([center_image, left_image, right_image, center_image_flp, left_image_flp, right_image_flp])
                angles.extend([center_steer, left_steer, right_steer, center_steer_flp, left_steer_flp, right_steer_flp])

            X_train = np.array(images)
            y_train = np.array(angles)

            # Shuffling again so to overcome the repeat due to data augmentation
            yield shuffle(X_train, y_train)

def resize_image(x, shape, scale):
    from keras.backend import tf as ktf
    dim = (shape[1] // scale, shape[0] // scale)
    return ktf.image.resize_images(x, dim)

def show_plot(history_object):
    import matplotlib.pyplot as plt

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def train_model(gen_train, hp, gen_valid):

    # Input
    inp = Input(hp.input_shape, name="input")

    # Cropping
    out = Cropping2D(cropping=hp.crop_area, name="cropping")(inp)

    # Scaling
    out = Lambda(resize_image, arguments={'shape':hp.input_shape, 'scale':hp.scale}, name='scaling')(out)

    # Normalization
    out = Lambda(lambda x: x/127.5 - 1.0, name="normalization")(out)

    #  Convolution
    out = Convolution2D(24, 5,5, activation='relu', name="convo1")(out)

    # Convolution
    out = Convolution2D(36, 5, 5, activation='relu', name="convo2")(out)

    # Convolution
    out = Convolution2D(48, 5, 5, activation='relu', name="convo3")(out)

    # Convolution
    out = Convolution2D(64, 3, 3, activation='relu', name="convo4")(out)

    # Convolution
    out = Convolution2D(64, 3, 3, activation='relu', name="convo5")(out)

    # Flatten
    out = Flatten(name="flatten")(out)

    # Fully-connected layer
    out = Dense(100, name="fully1")(out)

    # Fully-connected
    out = Dense(50, name="fully2")(out)

    # Fully-connected
    out = Dense(1, name="fully3")(out)

    model = Model(inp, out)

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(gen_train,
                                         nb_epoch=hp.epochs,
                                         samples_per_epoch=hp.num_samples,
                                         nb_val_samples=hp.num_samples,
                                         validation_data=gen_valid,
                                         verbose=1)

    # Save model to disk
    model.save('model.h5')

    # Show plot to visualize the loss
    show_plot(history_object)

def main():
    # Step 0: Init hyper-parameters
    hp = HyperParameters()

    # Step 1 : Extracting the required data
    samples_train, samples_valid = extract_samples()
    gen_train = batch_generator(samples_train, hp.batch_size, hp.correction)
    gen_valid = batch_generator(samples_valid, hp.batch_size, hp.correction)

    # Step 2: Set other Hyper-parameters
    hp.num_samples = len(samples_train) * 6
    hp.input_shape = read_image(samples_train[0][0]).shape

    # Step 3 : Train the model
    train_model(gen_train, hp, gen_valid)



if __name__ == '__main__':
    main()




