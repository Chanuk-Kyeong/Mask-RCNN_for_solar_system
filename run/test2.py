# import sys
# sys.path.append("/mrcnn")
from mrcnn.m_rcnn import *
import cv2
from mrcnn.visualize import random_colors, get_mask_contours, draw_mask


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # -1 only CPU / 0 use GPU0 

images_path = "dataset.zip"
annotations_path = "annotations.json"

# Load Image
# img = cv2.imread("dataset/02.jpg")
extract_images(os.path.join("./",images_path), "./dataset")
dataset_val = load_image_dataset(os.path.join("./", annotations_path), "./dataset", "val")

test_model, inference_config = load_inference_model(1, "logs/object20220818T1004/mask_rcnn_object_0005.h5")

test_random_image(test_model, dataset_val, inference_config)



# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Detect results
# r = test_model.detect([image])[0]
# colors = random_colors(80)


# # Get Coordinates and show it on the image
# object_count = len(r["class_ids"])
# for i in range(object_count):
#     # 1. Mask
#     mask = r["masks"][:, :, i]
#     contours = get_mask_contours(mask)
#     for cnt in contours:
#         cv2.polylines(img, [cnt], True, colors[i], 2)
#         img = draw_mask(img, [cnt], colors[i])
# cv2.imshow(img)





