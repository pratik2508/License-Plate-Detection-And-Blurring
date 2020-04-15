# License-plate-detection
This project using yolov3  to detection license plate in street images and blur then to hide user identity.
using repo : https://github.com/ultralytics/yolov3

# Description 
Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.0.0`
- `opencv-python`
- `tqdm
-  redis server for celery functioning
-  Flask for api creation 

`
# Data
- Training Data :47000 images (size 1480x720) 
- Test Data :  11000 images
- images include 3 types Front(F), Left(L), Right(R).
 
# How to training 
- Start Training: Run train.py --cfg cfg/yolov3.cfg --img-size 416

# Inference 
- Run `detect.py` to apply trained weights to an image, such as `car.jpg` from the `test` folder.

- Link weight  : 


# Test
- - Use `test.py --weights weights/latest.pt` to test YOLOv3 weights.

# Model has accuracy around 85%-90%  with no to very less false -ve's.  



# Wrapper use to detect and blur multiple images using postgres Database 

- we have created a wrapper class 'table_architecture' using postgreSQL and runs continously and once it finds a task of License plate it starts executing all images of that path.

- for License plate detection and multiple task management we are using celery.

- First coordinates of license plates in images are stored in database in x1,y1  x2,y2 format using detect function in 'detect_api2.py'.

- Then we will blur images as per blur api request in 'detect_api2.py'.

- a stop api has also been created for stopping execution in detect_api2.py'. 



