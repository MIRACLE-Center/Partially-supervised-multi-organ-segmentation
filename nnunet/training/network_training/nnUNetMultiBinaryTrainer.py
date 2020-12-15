from nnunet.training.network_training.nnUNetMultiTrainierV2 import nnUNetMultiTrainerV2
# from nnunet.training.dataloading.dataset_loading import DataLoader3DwithTag as DataLoader3D
from nnunet.training.dataloading.dataset_loading import DataLoader3DmergeTag as DataLoader3D
from nnunet.training.dataloading.dataset_loading import DataLoader2DwithTag as DataLoader2D
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

class nnUNetMultiBinaryTrainer(nnUNetMultiTrainerV2):
    def __init__(self, plans_file, fold, tasks,tags, output_folder_dict=None, dataset_directory_dict=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, tasks,tags, output_folder_dict=output_folder_dict, dataset_directory_dict=dataset_directory_dict, batch_dice=batch_dice, stage=stage,
                 unpack_data=unpack_data, deterministic=deterministic, fp16=fp16)
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

    def get_basic_generators(self, task):
        self.load_dataset(task)
        self.do_split(task)

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,self.tags[task],self.tags[self.tasks[0]],
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,self.tags[task],self.tags[self.tasks[0]], False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,self.tags[task],
                                 # self.plans.get('transpose_forward'),
                                 transpose=None,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,self.tags[task],
                                  # self.plans.get('transpose_forward'),
                                  transpose=None,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val