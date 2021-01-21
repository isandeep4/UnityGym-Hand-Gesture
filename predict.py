
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import operator
import cv2
import socket


UDP_IP = "127.0.0.1"
UDP_PORT = 5065
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

labels_path = 'labels.txt'
classes = []
#line = labels_file.readlines()
with open(labels_path, 'r') as f:
    classes = [line.split(' ',1)[1].strip() for line in f]
print(classes)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    #image = Image.open('image_1.jpg')
    #print(image)
    #flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flip = cv2.flip(frame,1)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    #image = ImageOps.fit(flip_image, size, Image.ANTIALIAS)
    img = cv2.resize(frame, size)
    #turn the image into a numpy array
    #image_array = np.asarray(image)
    
    # display the resized image
    #image.show()
    
    # Normalize the image
    normalized_image_array = (img.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # run the inference
    prediction = model.predict(data)
    print(prediction)
    predictions = {'JUMP': prediction[0][0], 
                      'UPPERCUT': prediction[0][1], 
                      'KICK': prediction[0][2],
                      'PLANK': prediction[0][3],
                      'NOTHING': prediction[0][4]
                      }
    predictions = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)
    cv2.putText(frame, predictions[0][0],(10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.imshow('Sign', frame)
    if predictions[0][0] == "JUMP":
        sock.sendto(("JUMP!").encode(), (UDP_IP, UDP_PORT))
        #print("_"*10, "kick Action Triggered!", "_"*10)
    elif predictions[0][0] == "UPPERCUT":
        sock.sendto(("Uppercut!").encode(), (UDP_IP, UDP_PORT))
        #print("_"*10, "uppercut Action Triggered!", "_"*10)
    elif predictions[0][0] == "KICK":
        sock.sendto(("Kick!").encode(), (UDP_IP, UDP_PORT))
    elif predictions[0][0] == "PLANK":
        sock.sendto(("Plank!").encode(), (UDP_IP, UDP_PORT))
    elif predictions[0][0] == "NOTHING":
        sock.sendto(("Nothing!").encode(), (UDP_IP, UDP_PORT))
    
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

