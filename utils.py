import os
import numpy as np
from torch.utils.data import DataLoader


def write_result_img(experiment_name, filename, img):
    root_path = '/image_results/'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)
    


def get_mgrid(data_shape):
    x = np.linspace(0, data_shape[0] - 1, data_shape[0])
    y = np.linspace(0, data_shape[1] - 1, data_shape[1])
    z = np.linspace(0, data_shape[2] - 1, data_shape[2])
    
    coordx, coordy, coordz = np.meshgrid(x, y, z)
    coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))

    return coord

                            
class DataWrapper(DataLoader):
    def __init__(self, dataset, data_shape, gt, compute_diff=None):
        self.dataset = dataset
        self.compute_diff = compute_diff
        self.mgrid = get_mgrid(data_shape)
        self.gt = gt

        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.mgrid}
    
        return in_dict, self.gt