Original File is located at:
https://colab.research.google.com/drive/1nxnStRzhppMBujUgTRAs1cQsZyZu7v_S#scrollTo=lR6JYta1JWq8

CODE:

# install dependencies: (use cu101 because colab has CUDA 10.1)
# opencv is pre-installed on colab
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
 
!gcc --version
 
# install detectron2:
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from google.colab import drive
drive.mount('/content/drive',force_remount=True)

!rm -r frames/*                  # removes any other frames present 
!mkdir frames/                   # "mkdir" command in Linux allows the user to create directories (also referred to as folders in some operating systems )
print(cv2.__version__)
vidcap = cv2.VideoCapture("/content/drive/My Drive/video/project_video.mp4")
# video="project_video.mp4"
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite('frames/'+str(count)+'.png', image)     # save frame as png file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
print(count)    

#number of frames in the given video
print("Number of frames in the given video = ",count)

cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
predictor = DefaultPredictor(cfg)

img = cv2.imread("frames/1.png")
cv2_imshow(img)

outputs = predictor(img)

v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
cv2_imshow(v.get_image())

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml')
predictor = DefaultPredictor(cfg)

#read an image
img = cv2.imread("frames/1.png")

#pass to the model
outputs = predictor(img)

v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img=cv2_imshow(v.get_image())

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


#read an image
img = cv2.imread("frames/1.png")

#pass to the model
outputs = predictor(img)

#read an image
img = cv2.imread("frames/1.png")

#pass to the model
outputs = predictor(img)

print(cfg.DATASETS)

print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

bounding_box=outputs['instances'].pred_boxes.tensor.cpu().numpy()  #converting tensor to numpy array 
classes=outputs['instances'].pred_classes.cpu().numpy() 
print(bounding_box)
print(classes)
print(len(classes))   #number of objects present in image

#identifying only "person" in a particular frame

index_of_person=np.where(classes==0)[0]
person=bounding_box[index_of_person]
num= len(person)
print("total number of people in a particular frame = ",num)

img = cv2.imread('frames/1.png')
v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img=v.get_image()
#define a function which return the bottom center of every bbox
def bottom_centre(img,person,idx):
  #get the coordinates
  x1,y1,x2,y2 = person[idx]
  _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
 
  mid=(int((x1+x2)/2),int(y2))
  
  _ = cv2.circle(img, mid, 5, (0, 0, 255), -1)
  cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)
  
  return mid

midpoints = [bottom_centre(img,person,i) for i in range(len(person))]
plt.figure(figsize=(20,10))
plt.imshow(img)


from scipy.spatial import distance
def calculate_distance(midpoints,num):
  dist = np.empty(shape=(num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist

dist=calculate_distance(midpoints,num)

def closest_people(dist,num,thresh):
  p1=[]
  p2=[]
  d=[]
  for i in range(num):
    for j in range(i,num):
      if( (i!=j) & (dist[i][j]<=thresh)):
        p1.append(i)
        p2.append(j)
        d.append(dist[i][j])
  return p1,p2,d
thresh=100
p1,p2,d=closest_people(dist,num,thresh)

def change_bounding_box_color(img,person,p1,p2):
  very_close = np.unique(p1+p2)
  for i in very_close:
    x1,y1,x2,y2 = person[i]
    _ = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)  
  return img
image=change_bounding_box_color(img,person,p1,p2)


import os
import re

names=os.listdir('frames/')
names.sort(key=lambda f: int(re.sub('\D', '', f)))

def find_closest_people_in_frame(name,thresh):

  img = cv2.imread('frames/'+name)
  outputs = predictor(img)
  classes=outputs['instances'].pred_classes.cpu().numpy()
  bounding_box=outputs['instances'].pred_boxes.tensor.cpu().numpy()
  index_of_person = np.where(classes==0)[0]
  person=bounding_box[index_of_person]
  v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
  v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
  img=v.get_image()
  midpoints = [bottom_centre(img,person,i) for i in range(len(person))]
  num = len(midpoints)
  dist= calculate_distance(midpoints,num)
  p1,p2,d=closest_people(dist,num,thresh)
  image = change_bounding_box_color(img,person,p1,p2)
  
  cv2.imwrite('frames/'+name,img)
  return 0
  
  
from tqdm import tqdm
thresh=100
_ = [find_closest_people_in_frame(names[i],thresh) for i in tqdm(range(len(names))) ]

frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

frame_array=[]

for i in range(len(frames)):
    
    #reading each files
    img = cv2.imread('frames/'+frames[i])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter('project_video_output-final_.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
