
import sys
from pathlib import Path
# sys.path.append("mrcnn_demo")
from mrcnn_demo.m_rcnn import *
from mrcnn_demo.visualize import *

model_name= 'solar_potential_25cm_final'
annotations_path = "annotations/{}_test_annotations.json".format(model_name)
# annotations_path = "annotations/{}_test_annotations.json".format(model_name)
print("Annotation")

dataset_path = 'dataset/{}_test/images'.format(model_name)
# dataset_path = 'dataset/{}_test/images'.format(model_name)

RATIO_M = 0.12 # 25cm 급->0.25, 12cm 급 -> 0.12
dataset_all = load_image_dataset(os.path.join("./", annotations_path), dataset_path, "all")
print(dataset_all.image_ids)
test_model, inference_config = load_inference_model(3, "model/{}_mask.h5".format(model_name))
# test_model, inference_config = load_inference_model(, "{}".format(model_name))
image_info = dataset_all.image_info

for image_id in dataset_all.image_ids:
    
    IMAGE_PATH= Path(image_info[image_id]['path'])
    image_name = IMAGE_PATH.name
    
    try:
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_all, inference_config,
                            image_id, use_mini_mask=False)
        image_path = 'gt_images/{}_test_gt_images/{}'.format(model_name, image_name)    
        # image_path = 'gt_images/{}_test_gt_images/{}'.format(model_name, image_name)
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                        dataset_all.class_names, figsize=(8, 8),path=image_path,RATIO_M=RATIO_M)
    
    except Exception as e:
        print(e)
        print("{} 예측 실패".format(image_name))
    

