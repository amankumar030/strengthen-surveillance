

#importing packages
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D,Input
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model,Model
from keras.optimizers import Adam
from collections import deque
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
import numpy as np
import os.path
from keras.preprocessing import image as Img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import glob
import os
import cv2
import math
import tensorflow as tf
import sys
from PIL import Image
from skimage.measure import compare_ssim as ssim
import shutil

VIDEO_NAME = 'test_video.mp4'


sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
#deleting presaved file

for file in os.listdir('croped_image'):
    filename = os.fsdecode(file)
    os.remove('croped_image/'+filename)
for file in os.listdir('collection'):
    filename = os.fsdecode(file)
    shutil.rmtree('collection/'+filename)
for file in os.listdir('video'):
    filename = os.fsdecode(file)
    os.remove('video/'+filename)

#loading saved model

model = Sequential()
model.add(LSTM(2048, return_sequences=False,input_shape=(10,2048),dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.load_weights('HAR/checkpoints/lstm-features.100-0.115.hdf5')

def rescale_list(input_list, size):
    assert len(input_list) >= size
    skip = len(input_list) // size
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]

def predict(x):
    image_name =str(x)
    cam = cv2.VideoCapture("video/"+image_name) 
    currentframe = 0
    classes=['help', 'non-help']
    frames=[]
    while(True): 
        ret,frame = cam.read() 
        if ret: 
        # if video is still left continue creating images 
            name = 'HAR/testFinal/frame'+image_name +"frame_no"+ str(currentframe) + '.jpg'
            cv2.imwrite(name, frame) 
            frames.append(name)  
            currentframe += 1
        else: 
            break
    cam.release() 
    cv2.destroyAllWindows() 
    rescaled_list = rescale_list(frames,10)

    base_model = InceptionV3(
    weights='imagenet',
    include_top=True
    )
# We'll extract features at the final pool layer.
    inception_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
    )
    sequence = []
    for image in rescaled_list:
        img = Img.load_img(image, target_size=(299, 299))
        x = Img.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = inception_model.predict(x)
        sequence.append(features[0])

    sequence = np.array([sequence])
    prediction = model.predict(sequence)
    maxm = prediction[0][0]
    maxid = 0
    for i in range(len(prediction[0])):
        if(maxm<prediction[0][i]):
            maxm = prediction[0][i]
            maxid = i

    print(classes[maxid])
    for file in os.listdir('HAR/testFinal'):
        filename = os.fsdecode(file)
        os.remove('HAR/testFinal/'+filename)
    return classes[maxid]

#standard dimension for croping
std_height=213
std_width=224
global fmt
fmt='.jpg'

def crop_and_save(frame,coordinates,count):
    img=frame
    num_of_detected_image=len(coordinates)
    for j in range(num_of_detected_image):
        croped=img[coordinates[j][0]:coordinates[j][1],coordinates[j][2]:coordinates[j][3]]
        croped_numpy=np.array(croped)
        croped_numpy= cv2.resize(croped_numpy, dsize=(std_height, std_width), interpolation=cv2.INTER_LINEAR)
        croped= Image.fromarray(croped_numpy, 'RGB')
        croped.save("croped_image/frame"+str(count)+"croped"+str(j)+fmt)

def MSE(imageA , imageB):
    err=np.sum((imageA.astype("float")- imageB.astype("float"))**2)
    err/=float(imageA.shape[0]*imageB.shape[1])
    return math.sqrt(err)

def compare_image(imageA,imageB):
    imageA=cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
    imageB=cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
    m=MSE(imageA , imageB)
    return m

MODEL_NAME = 'inference_graph'
video_count=1
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
NUM_CLASSES = 2
count=0
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()  #here1
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  #here2
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)   #here3


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
video = cv2.VideoCapture(PATH_TO_VIDEO)


for file in os.listdir('pred1'):
    filename = os.fsdecode(file)
    os.remove('pred1/'+filename)
for file in os.listdir('croped_image'):
    filename = os.fsdecode(file)
    os.remove('croped_image/'+filename)
for file in os.listdir('collection'):
    filename = os.fsdecode(file)
    shutil.rmtree('collection/'+filename)
    
    
while(video.isOpened()):
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    coordinates = vis_util.return_coordinates(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.80)
    for i in range(len(coordinates)):
        crop_and_save(frame,coordinates,count)
   # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)
       # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('HUMAN DETECTOR', frame)
    cv2.imwrite("pred1/image"+str(count)+".jpg", frame)
    
    count=count+1
    
    if(count==60):
        count=0
        #stacking
        images=[]
        
        for img in glob.glob("croped_image/*.jpg"):
            img_cv= cv2.imread(img)
            images.append(np.array(img_cv))
        print("croped images=", len(images))
        img_count=0
        stack_count=0
        
            
        if(len(images)!=0):
            
            directory = os.fsdecode('collection')
            cv2.imwrite(directory+"/"+"stack"+str(stack_count)+"/"+str(img_count)+".jpg",images[0])
            img_count=img_count+1
            stack_count=stack_count+1
            check=0
            for img in images:
                min_err=10000
                print("stacking image ...",check)
                check+=1 
                flag=0
                for stack in os.listdir(directory):
                    for file in os.listdir(directory+"/"+stack):
                        filename = os.fsdecode(file)
                        if(filename.endswith('.jpg')):
                            img_in_filename=Image.open(directory+"/"+stack+"/"+filename)
                            mse=compare_image(img,np.float32(img_in_filename))
                            if(mse<=45):
                                flag=1
                                if(min_err>mse):
                                    min_err=mse
                                    save=stack
                                   
                if(flag==1):
                    cv2.imwrite(directory+"/"+save+"/"+str(img_count)+".jpg",img)
                    img_count=img_count+1
                   
                if(flag==0):
                    stack_count=stack_count+1
                    os.mkdir(directory+"/"+"stack"+str(stack_count))
                    cv2.imwrite(directory+"/"+"stack"+str(stack_count)+"/"+str(img_count)+".jpg",img)
                    img_count=img_count+1
            
            #
            
            #delete the stack hving less then 10 frame       
            for stack in os.listdir(directory):
                no_of_frames=0
                for file in os.listdir(directory+"/"+stack):
                    if(no_of_frames<10):
                        no_of_frames=no_of_frames+1
                if(no_of_frames<10):
                    shutil.rmtree(directory+"/"+stack)
        
            #video
            directory = os.fsdecode('collection')
            #video_count=1
            
            for stack in os.listdir(directory):
               
            
                img_array = []
                for filename in glob.glob(directory+"/"+stack+"/*.jpg"):
                    img = cv2.imread(filename)
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
             
             
                   
                out = cv2.VideoWriter('video/'+str(video_count)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                video_count=video_count+1
                print("NUMBER OF VIDEO GENERATED : ",video_count-1)
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
             #code for prediction
            for file in os.listdir('video'):
                filename = os.fsdecode(file)
                
                if(predict(filename)=='help'):
                    source='video/'+filename
                    destination='help/'+filename
                    shutil.move(source, destination)
                else:
                    source='video/'+filename
                    destination='non-help/'+filename
                    shutil.move(source, destination)
                    
            
            
            
            
            
            
            #deleting the files
            for file in os.listdir('croped_image'):
                filename = os.fsdecode(file)
                os.remove('croped_image/'+filename)
            for file in os.listdir('collection'):
                filename = os.fsdecode(file)
                shutil.rmtree('collection/'+filename)
            for file in os.listdir('video'):
                filename = os.fsdecode(file)
                os.remove('video/'+filename)

            
        
   
   

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
