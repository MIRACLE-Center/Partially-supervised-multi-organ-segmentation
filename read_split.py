import os

path = '/home1/glshi/experiment/ckpt/TMI/checkpoint/singleV2/3d_fullres/'
subpath = 'nnUNetMultiTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw'
tasks = ['Task100_MALB','Task101_Liver','Task102_Spleen','Task103_Pancreas','Task104_KiTS']
for task in tasks:
    total_path = os.path.join(path,task,subpath)
    files = [i for i in os.listdir(total_path) if 'nii.gz' in i]
    print(files)