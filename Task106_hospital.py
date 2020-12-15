import os
import glob
import shutil
from collections import OrderedDict
import json
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk

if __name__ == "__main__":
    root = '/home1/glshi/data/private/hospitla'
    samples = os.listdir(root)
    output_dir = '/home1/glshi/data/nnUNet_preprocess/nnUNet_raw/Task106_hospital'
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
    output_test_img_files = []
    for i,sample in enumerate(samples):
        sub_sample = ""
        sample_path = os.path.join(root,sample)
        if os.path.isdir(sample_path):
            subdirs = os.listdir(sample_path)
            for subdir in subdirs:
                subdir = os.path.join(sample_path,subdir)
                print(subdir)
                if os.path.isdir(subdir):
                    if len(os.listdir(subdir))>=2:
                        sub_sample = subdir
        if not sub_sample=="":
            reader=sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(sub_sample)
            filenames=reader.GetGDCMSeriesFileNames(sub_sample,series_ids[0])
            reader.SetFileNames( filenames )
            sitk_img=reader.Execute()
            outname = f'case_{i:03}.nii.gz'
            output_test_img_files.append(os.path.join(output_Ts_img,outname))
            sitk.WriteImage(sitk_img, output_test_img_files[-1])
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
    json_dict['numTraining'] = 0
    json_dict['numTest'] = len(output_test_img_files)
    json_dict['training'] = []
    json_dict['test'] = [img_file for img_file in output_test_img_files]
    save_json(json_dict, join(output_dir, "dataset.json"))