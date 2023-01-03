# import sys
# sys.path.append("/mrcnn")
from mrcnn_demo.m_rcnn import *
import cv2
import pandas as pd

import os 





def test_val_image(test_model, dataset_val, inference_config, model_name, RATIO_M):
    print("compute mAP")
    
    # Compute VOC-style Average Precision
    # Pick a set of random images
    image_ids = dataset_val.image_ids
    image_info = dataset_val.image_info
    APs = []
    os.mkdir('area_images/{}_test_inference_images'.format(model_name))
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
        image_name = image_info[image_id]['path']
        image_name = image_name.split('\\')
        image_name = image_name[-1]
        results = test_model.detect([image], verbose=0)
        
        r = results[0]
        
        save_path = 'area_images/{}_test_inference_images/{}'.format(model_name, image_name)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax(), show_bbox=False, path=save_path,RATIO_M=RATIO_M)

        # Compute AP
        AP, precisions, recalls, overlaps ,dice_score=\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.7)
        APs.append(AP)
      except :
          print("{} 예측 실패".format(image_name))


    print("mAP @ IoU=70: ", np.mean(APs))
    
def Compute_performance(test_model, dataset_val, inference_config):
    
    # Compute VOC-style Average Precision
    # Pick a set of random images
    image_ids = dataset_val.image_ids
    image_info = dataset_val.image_info
    APs = []
    Precisions =[]
    Recalls =[]
    Dice_scores =[]
    image_names=[]
    TPs=[]
    FPs=[]
    FNs=[]
    for image_id in image_ids:
        # Load image
        try:
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config,
                                        image_id, use_mini_mask=False)
            # cv2.imwrite('/content/drive/MyDrive/태양광 사업/적지분석/데이터/항공이미지_25cm/테스트결과(삭제예정)/{}_gt.jpg'.format(image_id),img_gt)
            # Run object detection
            results = test_model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            gt_match, pred_match, overlaps = utils.compute_matches(
                gt_bbox, gt_class_id, gt_mask,
                                    r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.7)
            
            AP, precisions, recalls, overlaps ,dice_score=\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                    r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.7)
                
            image_name = image_info[image_id]['path']
            image_name = image_name.split('\\')
            image_name = image_name[-1]
            
            TP = round((len(gt_match))*recalls[-2])
            FP = len(pred_match) - TP
            FN = len(gt_match) - TP
            
            # FP = round((len(pred_match) + 1) - TP)
            
            image_names.append(image_name)
            APs.append(AP)
            Precisions.append(np.mean(precisions[-2]))
            Recalls.append(np.mean(recalls[-2]))
            Dice_scores.append(np.mean(dice_score[-1]))
            TPs.append(TP)
            FPs.append(FP)
            FNs.append(FN)
        except :
          print("{} 예측 실패".format(image_name))
   

    # print("images:", image_names)
    # print("APs @ IoU=70: ", APs)
    # print("precisions @ IoU=70: ", Precisions)
    # print("recalls @ IoU=70: ", Recalls)
    # print("dice_scores @ IoU=70: ",Dice_scores)
    # print("TPs: ",TPs)
    # print("FPs: ",FPs)
    # print("FNs: ",FNs)
    
    return image_names, TPs, FPs, FNs, Precisions, Recalls, APs, Dice_scores

if __name__=='__main__':
    RATIO_M = 0.25 # 25cm 급->0.25, 12cm 급 -> 0.12
    
    
    
    #원하는 모델
    model_name = 'solar_potential_25cm_final'
    

    #model_path
    mask_model_path = "model/{}_mask.h5".format(model_name)
    #load model
    test_model, inference_config = load_inference_model(1, mask_model_path)
    
    #extract test images
    images_path = "dataset/{}_test/images".format(model_name)
    annotations_path = "annotations/{}_test_annotations.json".format(model_name)
    dataset_all = load_image_dataset(os.path.join("", annotations_path), images_path, "all")
    
    image_names, TPs, FPs, FNs, Precisions, Recalls, APs, Dice_scores = Compute_performance(test_model, dataset_all,inference_config)
    data = {
        'image_names': image_names,
        'TPs': TPs,
        'FPs': FPs,
        'FNs': FNs,
        'Precisions':Precisions,
        'Recalls':Recalls, 
        'APs': APs,
        'Dice_scores': Dice_scores
    }
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('유효성 지표별 제출로그 양식({})_mask r cnn.csv'.format(model_name),index=False)