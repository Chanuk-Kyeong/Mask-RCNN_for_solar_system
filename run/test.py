# import sys
# sys.path.append("/mrcnn")
from mrcnn_demo.m_rcnn import *
import cv2
import os
from pathlib import Path


def show_mask(test_model, dataset_val, inference_config):
    print("compute mAP")

    # Compute VOC-style Average Precision
    # Pick a set of random images
    image_ids = dataset_val.image_ids
    APs = []
    for image_id in image_ids:
        # Load image
      try:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)

        # image, gt_bbox, gt_mask, gt_class_id,
        #                     dataset_val.class_names)
        # cv2.imwrite('/content/drive/MyDrive/태양광 사업/적지분석/데이터/항공이미지_25cm/테스트결과(삭제예정)/{}_gt.jpg'.format(image_id),img_gt)
        # Run object detection

        results = test_model.detect([image], verbose=0)

        r = results[0]

        visualize.display_top_masks(
            image, r['masks'], r['class_ids'], dataset_val.class_names)

      except ZeroDivisionError as e:
          print("{}번째 figure 예측 실패".format(image_id))


def test_val_image(test_model, dataset_val, inference_config, model_name, RATIO_M):
    print("compute mAP")

    # Compute VOC-style Average Precision
    # Pick a set of random images
    image_ids = dataset_val.image_ids
    image_info = dataset_val.image_info
    APs = []
    os.mkdir('area_imagess/{}_test_inference_images'.format(model_name))
    for image_id in image_ids:
        # Load image
      try:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)

        # img_gt = visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
        #                     dataset_val.class_names)
        # cv2.imwrite('/content/drive/MyDrive/태양광 사업/적지분석/데이터/항공이미지_25cm/테스트결과(삭제예정)/{}_gt.jpg'.format(image_id),img_gt)
        # Run object detection

        IMAGE_PATH = Path(image_info[image_id]['path'])
        image_name = IMAGE_PATH.name
        results = test_model.detect([image], verbose=0)

        r = results[0]

        # save_path = 'area_images/{}_test_inference_image_1/{}'.format(model_name, image_name)
        save_path = Path(f'area_imagess/{model_name}_test_inference_images')
        image_path = Path(image_name).name
        save_path = save_path / image_path
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_val.class_names, r['scores'], ax=get_ax(), show_bbox=False, path=save_path, RATIO_M=RATIO_M)

        # Compute AP
        AP, precisions, recalls, overlaps, dice_score =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.7)
        APs.append(AP)
      except Exception as e:
        print(e)
        print("{} 예측 실패".format(image_name))

    print("mAP @ IoU=70: ", np.mean(APs))


def Compute_performance(test_model, dataset_val, inference_config):

    # Compute VOC-style Average Precision
    # Pick a set of random images
    image_ids = dataset_val.image_ids
    APs = []
    Precisions = []
    Recalls = []
    Dice_scores = []
    try:
        for image_id in image_ids:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
            # cv2.imwrite('/content/drive/MyDrive/태양광 사업/적지분석/데이터/항공이미지_25cm/테스트결과(삭제예정)/{}_gt.jpg'.format(image_id),img_gt)
            # Run object detection
            results = test_model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            
            AP, precisions, recalls, overlaps, dice_score =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.7)

            APs.append(AP)

            Precisions.append(np.mean(precisions[-2]))
            Recalls.append(np.mean(recalls[-2]))
            Dice_scores.append(np.mean(dice_score[-1]))

    except Exception as e:
        print(e)

    mAP = np.mean(APs)
    mPrecisions = np.mean(Precisions)
    mRecalls = np.mean(Recalls)
    mDice_scores = np.mean(Dice_scores)
    print("mAP @ IoU=70: ", mAP)
    print("precision @ IoU=70: ", mPrecisions)
    print("recall @ IoU=70: ", mRecalls)
    print("dice_score @ IoU=70: ", mDice_scores)


if __name__ == '__main__':
    RATIO_M = 0.25  # 25cm 급->0.25, 12cm 급 -> 0.12

    #원하는 모델
    # model_name = 'solar_potential_25cm_8234_checked'
    model_name = 'solar_potential_25cm_final'
    # model_name = 'solar_potential_25cm_21975_total'

    #model_path
    mask_model_path = "model/{}_mask.h5".format(model_name)

    #extract test images
    images_path = "dataset/{}_test/images".format(model_name)
    annotations_path = "annotations/{}_test_annotations.json".format(
        model_name)
    dataset_test = load_image_dataset(os.path.join(
        "", annotations_path), images_path, "all")
    class_number = dataset_test.count_classes()
    #load model
    test_model, inference_config = load_inference_model(
        class_number, mask_model_path)

    test_val_image(test_model, dataset_test, inference_config, model_name, RATIO_M)
    # calcuate_Area(images_path, test_model, model_name)
    Compute_performance(test_model, dataset_test, inference_config)
