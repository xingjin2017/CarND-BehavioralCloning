import numpy as np
import math
import pandas as pd
import os
import cv2
import time
import random

from datetime import datetime

from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf
tf.python.control_flow_ops = tf

from sklearn.utils import shuffle

# The percentage of data to use - smaller number for quick firetest
use_data_ratio = 1.0
train_ratio = 1.0

data_root_dir = './data'

log_csv = pd.read_csv(os.path.join(data_root_dir, 'driving_log.csv'))

# Do we keep the order and use LSTM for this?
shuffled = log_csv.sample(frac=use_data_ratio)
num_rows = len(shuffled)
num_train = int(num_rows * train_ratio)
num_val = max(1000, num_rows - num_train)

'''
steering_labels = train_rows['steering'].values.astype('float32')
throttle_labels = train_rows['throttle'].values.astype('float32')
brake_labels = train_rows['brake'].values.astype('float32')

Y_training = np.dstack((steering_labels, throttle_labels))[0]

center_values = train_rows['center'].values.astype('str')
left_values = train_rows['left'].values.astype('str')
right_values = train_rows['right'].values.astype('str')

X_img_locations = np.dstack((center_values, left_values, right_values))[0]
'''

def chop_image(img):
    ''' original 160, 320, 3 => (112, 320, 3)
    '''
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chopped = img[32:144, :, :]
    return chopped

def normalize_image(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    mean_values = np.mean(img, axis=(0, 1))
    normalized = (img - mean_values) / 255
    return normalized

class ProcessedRowData(object):
    '''
    A container class that has decoded images and corresponding control
    numbers
    '''
    # dataframe.iloc[0]['center']
    def __init__(self, index, log_row):
        self.index = index
        img = self.extract_image(log_row['center'])
        self.center_img = chop_image(img)
        img = self.extract_image(log_row['left'])
        self.left_img = chop_image(img)
        img = self.extract_image(log_row['right'])
        self.right_img = chop_image(img)
        self.steering = log_row['steering']
        self.throttle = log_row['throttle']
        self.brake = log_row['brake']
        # speed around 30
        self.speed = log_row['speed'] / 30.0

    def extract_image(self, path):
        path = str(path)
        path = path.strip()
        if path == '' or path == 'nan':
            return None
        if not path.startswith('/'):
            path = os.path.join(data_root_dir, path)
        return cv2.imread(path)
        
    def report(self):
        print('Reporting on row original index: %d' % self.index)
        print('center_img shape:', self.center_img.shape)
        print('left_img shape:', self.left_img.shape)
        print('right_img shape:', self.right_img.shape)
        print('steering %.5f throttle %.5f brake %.5f speed %.5f' %
              (self.steering, self.throttle, self.brake,
              self.speed))
        print()

pt0 = time.time()        
processed_data = []
steerings = []
throttles = []
speeds = []
for idx, row in shuffled.iterrows():
    row = ProcessedRowData(idx, row)
    processed_data.append(row)
    steerings.append(row.steering)
    throttles.append(row.throttle)
    speeds.append(row.speed)
print("Processing data rows took %f seconds" % (time.time() - pt0))
print("Steering: mean %.5f min %.5f max %.5f" %
      (np.mean(steerings), np.min(steerings), np.max(steerings)))
print("Throttle: mean %.5f min %.5f max %.5f" %
      (np.mean(throttles), np.min(throttles), np.max(throttles)))
print("Speed: mean %.5f min %.5f max %.5f" %
      (np.mean(speeds), np.min(speeds), np.max(speeds)))
print("processed_data has %d elements" % len(processed_data))

processed_train_data = processed_data[0:num_train]
processed_val_data = processed_data[-num_val:]

# Assume the final activation is tanh -1 to 1
mean_throttle = 0.87
def normalize_control(steering, throttle):
    # steering -1 to 1, mean 0; throttle 0 to 1, mean 0.87
    n_steering = steering / 10.0
    n_throttle = (throttle - mean_throttle) / 10.0
    return n_steering, n_throttle

def denormalize_control(n_steering, n_throttle):
    steering = 10.0*n_steering
    throttle = 10.0*n_throttle + mean_throttle

random.seed(datetime.now())

def adjust_image_brightness(img):
    new_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    adjust_ratio = 0.2 + np.random.uniform()
    new_image[:, :, 2] = new_image[:, :, 2]*adjust_ratio
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
    return new_image

# move image in x and y directions
def shake_image(img, steering, throttle, x_move_range,
                y_move_range):
    delta_x = x_move_range*(np.random.uniform() - 0.5)
    delta_y = y_move_range*(np.random.uniform() - 0.5)
    translation_matrix = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    num_rows, num_cols = img.shape[:2]
    img_translation = cv2.warpAffine(
        img, translation_matrix, (num_cols, num_rows))
    new_steering = steering + delta_x / x_move_range * 0.2
    new_throttle = max(0.2, throttle - abs(delta_x)*0.01)
    return img_translation, new_steering, new_throttle

def test_augmentation():
    img = processed_data[0].center_img
    cv2.imwrite('/home/xjin/test/org.jpg', img)
    for i in range(5):
        new_img, _, _ = shake_image(img, 0, 0, 320, 0)
        fname = '/home/xjin/test/shake{}.jpg'.format(i+1)
        cv2.imwrite(fname, new_img)
        new_img = adjust_image_brightness(img)
        fname = '/home/xjin/test/bright{}.jpg'.format(i+1)
        cv2.imwrite(fname, new_img)
        
# test_augmentation()

def generate_data_for_one_row(row, disable_augmentation):
    # case = np.random.randint(4)
    case = np.random.uniform()
    img = row.center_img
    speed = row.speed
    steering = row.steering
    throttle = row.throttle
    using_center = True

    if disable_augmentation:
        return normalize_image(img), speed, steering, throttle
    
    if case <= 0.25:
        pass
    elif case <= 0.5:
        # flip the image
        img = cv2.flip(row.center_img, 1)
        steering = -row.steering
    elif case <= 0.75:
        using_center = False
        img = row.left_img
        steering = row.steering + 0.3 # 0.25
        throttle = row.throttle / 2.0
    else:
        using_center = False
        img = row.right_img
        steering = row.steering - 0.3
        # throttle = max((row.throttle - 0.4), 0.2)
        throttle = row.throttle / 2.0

    # Adjust image brightness
    case = np.random.uniform()
    if case <= 1.0:
        img = adjust_image_brightness(img)

    # move image around
    x_move_range = 150
    y_move_range = 40
    case = np.random.uniform()
    if case <= 1.0:
        img, steering, throttle = shake_image(
            img, steering, throttle, x_move_range, y_move_range)

    img = normalize_image(img)
    return img, speed, steering, throttle

def generate_data(processed_data_rows, angle_threshold,
                  low_angle_keep_ratio, batch_size,
                  disable_augmentation=False):
    num_total = len(processed_data_rows)
    while True:
        X_input = np.empty((batch_size, 64, 64, 3), dtype=np.float32)
        X_speed = np.empty((batch_size,), dtype=np.float32)
        Y_steering = np.empty((batch_size,), dtype=np.float32)
        Y_throttle = np.empty((batch_size,), dtype=np.float32)
        cnt = 0 # number of images had so far
        while cnt < batch_size:
            i = random.randint(0, num_total-1)
            row = processed_data_rows[i]
            img, speed, steering, throttle = generate_data_for_one_row(
                row, disable_augmentation)
            if img is None:
                continue
            if (abs(steering) < angle_threshold and
                np.random.uniform() > low_angle_keep_ratio):
                continue
            X_input[cnt] = img
            X_speed[cnt] = speed
            Y_steering[cnt] = steering
            Y_throttle[cnt] = throttle
            cnt += 1
            if cnt >= batch_size:
                break
        # yield ({'image_input': X_input, 'speed_input': X_speed},
        yield ({'image_input': X_input},
               {'steering': Y_steering, 'throttle': Y_throttle})

# Test the generator        
def test_generator():
    cnt = 0
    for x, y in generate_data(processed_train_data, 0.5, 0.0, 1):
        print(y['steering'])
        print()
        cnt += 1
        if cnt > 20:
            break

# test_generator()

def build_model():
    # model = Sequential()
    image_input = Input(shape=(64, 64, 3), name='image_input')
    #speed_input = Input(shape=(1,), name='speed_input')
    net = Convolution2D(3, 1, 1, border_mode='valid', init='he_normal')(image_input)
    net = Convolution2D(32, 3, 3, border_mode='valid', init='he_normal')(net)
    net = ELU(alpha=1.0)(net)
    net = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(net)
    net = Dropout(0.5)(net)
    net = Convolution2D(64, 3, 3, border_mode='valid', init='he_normal')(net)
    net = ELU(alpha=1.0)(net)
    net = Activation('relu')(net)
    net = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(net)
    net = Dropout(0.5)(net)
    net = Convolution2D(128, 3, 3, border_mode='valid', init='he_normal')(net)
    net = ELU(alpha=1.0)(net)
    net = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    # net = merge([net, speed_input], mode='concat')
    net = Dense(512, init='he_normal')(net)
    net = ELU(alpha=1.0)(net)
    net = Dropout(0.5)(net)
    net = Dense(64, init='he_normal')(net)
    net = ELU(alpha=1.0)(net)
    net = Dropout(0.5)(net)
    net = Dense(16, init='he_normal')(net)
    #net = LeakyReLU(alpha=0.1)(net)
    net = ELU()(net)
    net = Dropout(0.5)(net)
    # output two numbers: steering and throttle for the
    # drive function
    steering = Dense(1, name='steering')(net)
    #tnet = merge([net, speed_input], mode='concat')
    #tnet = Dense(12)(tnet)
    #tnet = LeakyReLU(alpha=0.1)(tnet)
    throttle = Dense(1, name='throttle')(net)
    model = Model(input=[image_input], output=[steering, throttle])
    loss_dict = {'steering': 'mean_squared_error', 'throttle': 'mean_squared_error'}
    model.compile(optimizer='adam', loss=loss_dict, loss_weights=[0.9, 0.1])
    return model

model = build_model()


one_batch_size = 256
val_generator = generate_data(processed_train_data, 0.0, 1.0,
                              one_batch_size, disable_augmentation=True)
angle_threshold = 0.0
low_angle_keep_ratio = 1.0

num_eps = 20
histories = []
ps = datetime.now().strftime(".%H_%M_%S")
for ep in range(num_eps):
    print("Episode %d using angle_threshold %.3f and low_angle_keep_ratio %.1f%%"
          % (ep+1, angle_threshold, low_angle_keep_ratio*100))
    train_generator = generate_data(processed_train_data, angle_threshold,
                                    low_angle_keep_ratio, one_batch_size)
    history = model.fit_generator(train_generator,
                              samples_per_epoch=80*one_batch_size, nb_epoch=1,
                              validation_data=val_generator,
                              nb_val_samples=10*one_batch_size)
    mv = 'model.h5-{:0>2}{}'.format((ep+1), ps)
    # model.save_weights(mv)
    histories.append((history.history, angle_threshold, low_angle_keep_ratio))
    angle_threshold += 0.05
    angle_threshold = min(0.4, angle_threshold)
    low_angle_keep_ratio -= 0.1
    low_angle_keep_ratio = max(0.3, low_angle_keep_ratio)

# ps = datetime.now().strftime(".%H-%M-%S")
with open("model.json"+ps, "w") as json_file:
    json_file.write(model.to_json())
model.save_weights('model.h5'+ps)
print("Model saved")

with open("train.history"+ps, "w") as hist_file:
    cnt = 1
    for history, angle_threshold, low_angle_keep_ratio in histories:
        hist_file.write("Episode {}: angle_threshold {} low_angle_keep_ratio {}"
                        .format(cnt, angle_threshold, low_angle_keep_ratio))
        hist_file.write("\n")
        hist_file.write(str(history))
        hist_file.write("\n\n")
        cnt += 1

