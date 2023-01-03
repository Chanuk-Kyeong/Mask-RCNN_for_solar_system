
import sys
# sys.path.append("mrcnn_demo")
from mrcnn_demo.m_rcnn import *
from mrcnn_demo.visualize import *

model_name='solar_potential_25cm_324_test'
annotations_path = "annotations/{}_annotations.json".format(model_name)
print("Annotation")
dataset_path = 'dataset/{}_images'.format(model_name)

# dataset_train = load_image_dataset(os.path.join("./", annotations_path), "./dataset/Images", "train")
# dataset_train = load_image_dataset(os.path.join("./", annotations_path), d`ataset_path, "train")
# dataset_val = load_image_dataset(os.path.join("./", annotations_path), dataset_path, "val")
dataset_all = load_image_dataset(os.path.join("./", annotations_path), dataset_path, "all")
print(dataset_all.image_ids)
# test_model, inference_config = load_inference_model(2, "logs/solar_potential_25cm_251_256x256_mask.h5")

# for image_id in dataset_all.image_ids:
#     original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#             modellib.load_image_gt(dataset_all, inference_config,
#                                 image_id, use_mini_mask=False)
#     image_path = 'gt_images/{}_gt_images/{}_gt.jpg'.format(model_name, image_id)
#     visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                                     dataset_all.class_names, figsize=(8, 8),path=image_path)
display_image_samples(dataset_all)
print(eree)

 