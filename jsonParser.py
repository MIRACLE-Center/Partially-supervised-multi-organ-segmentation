import json
import os
if __name__ == "__main__":
    task = 'Task104_KiTS'
    my_output_identifier = 'singleV2_10'
    trainer = 'nnUNetMultiTrainerV2__nnUNetPlansv2.1'
    # trainer = 'nnUNetTrainerV2__nnUNetPlansv2.1'
    json_root = os.path.join('/home1/glshi/experiment/ckpt/TMI/checkpoint/',my_output_identifier,'3d_fullres',task,trainer,'fold_4/validation_raw/')
    file = os.path.join(json_root,'summary.json')
    f = open(file,'r')
    result_dict = json.load(f)
    result = result_dict['results']['all']
    # print(result)
    dice_max = {}
    dice_min = {}
    hd_max = {}
    hd_min = {}
    for item in result:
        for key in item.keys():
            if len(key)>2:
                continue
            dice = float(item[key]['Dice'])
            hd = float(item[key]['Hausdorff Distance 95'])
            if key not in dice_max or dice_max[key]<dice:
                dice_max[key] = dice
            if key not in dice_min or dice_min[key]>dice:
                dice_min[key] = dice
            if key not in hd_max or hd_max[key]<hd:
                hd_max[key] = hd
            if key not in hd_min or hd_min[key]>dice:
                hd_min[key] = hd
    print(f'Now Reading {my_output_identifier}')
    mean = result = result_dict['results']['mean']
    for key in dice_max.keys():
        dice_mean = float(mean[key]['Dice'])
        hd_mean = float(mean[key]['Hausdorff Distance 95'])
        var_dice = max(dice_mean-dice_min[key],dice_max[key]-dice_mean)
        var_hd = max(hd_mean-hd_min[key],hd_max[key]-hd_mean)
        print(f'Key {key} range:')
        print(f'Dice:   [{dice_min[key]}]-[{dice_max[key]}]')
        print(f'HD:     [{hd_min[key]}]-[{hd_max[key]}]')
        print(f'Var:    [{var_dice}]/[{var_hd}]')

            

