from nnunet.utilities.Image import *
import numpy as np
import SimpleITK as sitk
import os
import glob

predtags = ['liver','spleen','pancreas','leftkidney','rightkidney']
labeltags = [predtags[0]]
# labeltags = predtags
# predtags = labeltags

if __name__ == "__main__":
    task = 'Task105_LiverChallenge'
    img_root = os.path.join('/home1/glshi/data/nnUNet_preprocess/nnUNet_raw_splitted/',task,'imagesTr/')
    trainer = 'nnUNetMultiTrainerV2__nnUNetPlansv2.1'
    # trainer = 'nnUNetTrainerV2__nnUNetPlansv2.1'
    pred_root = os.path.join('/home1/glshi/experiment/ckpt/TMI/checkpoint/',my_output_identifier,'3d_fullres',task,trainer,'fold_4/validation_raw/')
    label_root = os.path.join('/home1/glshi/experiment/ckpt/TMI/checkpoint/',my_output_identifier,'3d_fullres',task,trainer,'gt_niftis/')
    # img_path = '/home1/glshi/data/nnUNet_preprocess/final_data/Task100_MALB/nnUNetData_plans_v2.1_stage0/01_02.npy'
    files = os.listdir(pred_root)
    files = [file for file in files if (file.endswith('.nii.gz'))]
    for pred_name in files:
        fname = pred_name[:-7]
        im_name = fname+'_0000.nii'
        img_path = os.path.join(img_root,im_name)
        pred_path = os.path.join(pred_root,pred_name)
        label_path = os.path.join(label_root,pred_name)
        img_nii = sitk.ReadImage(img_path)
        pred_nii = sitk.ReadImage(pred_path)
        # label_nii = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(img_nii)
        pred = sitk.GetArrayFromImage(pred_nii)
        # label = sitk.GetArrayFromImage(label_nii)



        # image = np.transpose(image,[2,1,0])
        # pred = np.transpose(pred,[2,1,0])
        # label = np.transpose(label,[2,1,0])
        # pred[pred>0] = 0
        fname = os.path.join(task+'ori',fname)
        WritePrediction(image,pred,fname,predtags)
        # WritePredictionWithLabel(image,label,pred,fname,predtags,labeltags)