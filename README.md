# CarND-BehavioralCloning
CarND Project 3: behavioral cloning

# The architecture of the model

The model takes input the camera image and outputs a steering angle and a throttle value.

The initial layers are common CNN layers, with a leading 1x1 convolution to convert the input image color space. Followed by Conv2D/ELU activation/MaxPooling/Dropout layers. The feature map sizes are: 32x3x3 => 64x3x3 => 128x3x3. 

Then they are followed by fully connected layers (Dense/ELU activation/Dropout): 512 => 64 => 16.

The final outputs are two separate heads with the final fully connected layer (width 16) as input, one Dense(1) for steering output and another Dense(1) for throttle output.

Also attempted a version that had speed as one additional input, which was concatenated to the first FC (512) layer, but didn't follow it through to see if it would drive fully successfuly. It is tied to throttle in a way.

# Training methods and data augmentation

Used the dataset provided by Udacity as it is not easy to collect data with a mouse in the simulator. A bunch of data augmentation methods were used based on articles read on the web:

1. Cut the top and a small part of the bottom of the camera image off, so not to include the sky/scenary/car dashboard in the training.

2. Flip the image and reverse the sign of the steering angle, to double the amount of data.

3. Use the left camera image and the right camera image, but adjusted the target steering angles by an amount around 0.3 or 0.25 manually.

4. Translate camera image horizontially and vertically, and also adjust the steering angle slightly based on the horizontal distance moved. This would be useful if the car doesn't strictly stay in the center of the road.

5. Train about 20 epoches, with 20000 or so images used for each epoch. Have to use every image for training, because some sections of the road don't have much data, but are very important need to be covered in the training (say, a fork to the roadside).

# Results

The car can drive autonomously in the first track indefinitely in my simulator (the beta version). The command used to drive is:

python drive.py model.json

# Further thoughts

I had some fancier thoughts on the architecture of the network, say to integrate LSTM into the network, so the car can take the recent driving history and current state into account when making a steering/throttle decision. This would make sense and possibly prevent jerky motions from one frame to the next. Because the lack of time and training data, didn't follow this through.

Also, instead of hardcoding some steering number changes during data augmentation by shifting images around, it would probably be better to warp/project the camera image at an angle instead, so it would be more realistic that the car is viewing the scene at a particular angle.

During training, I found the architecture/data augmentation methods are very senstive to the hard coded numbers used, say, add 0.3 for using left camera image for simulation. Slight change of the numbers would make the driving fail and the car would wonder off the track somewhere for some unclear reason. In summary, it is fragile and very sensitive to the numbers being used.

Hopefully more data will help in these cases, but in general, probably some other methods probably are also needed to make the driving robust, and to make the car has a better understanding of the scene and the situation it is currently in.


