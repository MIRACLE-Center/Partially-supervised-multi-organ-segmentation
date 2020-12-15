import os
import glob
import shutil
from collections import OrderedDict
import json
from batchgenerators.utilities.file_and_folder_operations import *

if __name__ == "__main__":
    root = '/home1/glshi/data/private/challenge'
    samples = os.listdir(root)
    output_dir = '/home1/glshi/data/nnUNet_preprocess/nnUNet_raw/Task105_LiverChallenge'
    output_Tr_img = os.path.join(output_dir,'imagesTr')
    output_Tr_gt = os.path.join(output_dir,'labelsTr')
    output_Ts_img = os.path.join(output_dir,'imagesTs')
    if not os.path.exists(output_Tr_img):
        os.makedirs(output_Tr_img)
    if not os.path.exists(output_Tr_gt):
        os.makedirs(output_Tr_gt)
    if not os.path.exists(output_Ts_img):
        os.makedirs(output_Ts_img)
    output_train_img_files = []
    output_train_gt_files = []
    for i,sample in enumerate(samples):
        img_file = os.path.join(root,sample,'venous..nii.gz')
        gt_file = os.path.join(root,sample,'venous..ni_roi_wqs.nii')
        outname = f'case_{i:03}.nii.gz'
        output_train_img_files.append(os.path.join(output_Tr_img,outname))
        output_train_gt_files.append(os.path.join(output_Tr_gt,outname))
        shutil.copy(img_file,os.path.join(output_Tr_img,outname))
        shutil.copy(gt_file,os.path.join(output_Tr_gt,outname))
    json_dict = OrderedDict()
    json_dict['name'] = "Liver_Segmentation"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Liver"
    }
    json_dict['numTraining'] = len(output_train_img_files)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': img_file, "label": gt_file}
                             for img_file, gt_file in zip(output_train_img_files, output_train_gt_files)]
    json_dict['test'] = []
    save_json(json_dict, join(output_dir, "dataset.json"))