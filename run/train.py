import sys

# from .mrcnn.m_rcnn import MyConfig
# sys.path.append("mrcnn")
from mrcnn_demo.m_rcnn import *

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import os

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Extract Images
images_path_train = "dataset/solar_potential_25cm_final_train/images"    #이미지 폴더 경로 수정(훈련용:solar~.._train/images)
annotations_path_train  = "annotations/solar_potential_25cm_final_train_annotations.json"  #annoation 폴더 경로 수정(훈련용:solar~.._train)
images_path_val = "dataset/solar_potential_25cm_final_val/images" #이미지 폴더 경로 수정(검사용:solar~.._val/images)
annotations_path_val  = "annotations/solar_potential_25cm_final_val_annotations.json" #annoation 폴더 경로 수정(검사용:solar~..val)

# extract_images(os.path.join("./",images_path_train), "./dataset")

dataset_train = load_image_dataset(os.path.join("./", annotations_path_train), images_path_train, "all")
dataset_val = load_image_dataset(os.path.join("./", annotations_path_val), images_path_val, "all")

class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))

# Load image samples
# display_image_samples(dataset_train)

# Load Configuration
config = CustomConfig(class_number) 
config.NUM_CLASSES = class_number
config.ETF_C = class_number+1
config.IMAGE_META_SIZE= 12 +class_number+1

model = load_training_model(config)
model.config.IMAGE_META_SIZE= 12 +class_number+1

# Start Training
# This operation might take a long time. 
train_head(model, dataset_train, dataset_val, config)

