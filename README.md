# UnityGym-Hand-Gesture

* The project was created from scratch using Unity and it recognizes hand gestures in real time from camera input, used a deep neural network CNN and opencv to detect the hand movement and the relative gesture.
* It does this by detecting skin pixels by creating an HSV colour range in which only the skin pixels exist. Afterwards the image is separated into black and white (non-skin pixels and skin pixels, respectively) and is reduced to a 24 by 32 image, by segmenting the image into squares and finding their average.
* 32 by 32 image is given as 1D input to a neural network that decides what gesture is being shown. Training data for backpropagation is created by recording the user’s hand while they preform a gesture. This data is then saved onto the computer and can be used at any time to train the network.
