import os, glob
import numpy as np
from torch.utils.data import DataLoader
import torch


def detach_tensor(tensor_array, data_shape): 
    '''Takes in a tensor as input and returns a numpy array of shape data_shape'''

    if isinstance(tensor_array, np.ndarray):
        return tensor_array
    
    new_array = np.reshape(
            tensor_array.cpu().detach().numpy(), 
            (data_shape[0], data_shape[1], data_shape[2], -1)
        )
    return new_array


def attach_tensor(numpy_array):
    '''Takes in an array as input and returns a tensor of shape (1,numpy_array.shape[0]*numpy_array.shape[1]*numpy_array.shape[2],1)'''
    if torch.is_tensor(numpy_array):
        return numpy_array
    
    new_tensor = np.expand_dims(np.reshape(numpy_array, (-1, numpy_array.shape[-1])), axis=0)
    new_tensor = np.expand_dims(np.sum(new_tensor, -1), axis=-1)
    new_tensor = torch.from_numpy(new_tensor.astype(np.float32)).cuda()
    new_tensor.requires_grad = False

    return new_tensor


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
        self.data_shape = data_shape

        
    def __len__(self):
        return self.data_shape[0]
    
    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.mgrid}
    
        return in_dict, self.gt
    
    

def square_matrix(square):
    """ This function will calculate the value x
       (i.e. blurred pixel value) for each 3 * 3 blur image
       as the mean average of the neighboring pixels and the
       central pixel value.
    """
    tot_sum = 0

    # Calculate sum of all the pixels in 3 * 3 matrix
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                tot_sum += square[i][j]
                
    pixel_val = ((tot_sum // 8) + square[1][1]) // 2

    return pixel_val  


def concat_full_matrix(original, blur_img):
    """
    A helper function that combines the blurred image padded with
    the original image around the border. Returning the same shape
    as the original image.
    """
    averaged_img = []
    n_row = len(original)
    n_col = len(original[0])

    for i in range(0, n_row):
        for j in range(0, n_col):
            if i == 0 or i == n_row-1 or j == 0 or j == n_col-1:
                averaged_img.append(original[i][j])
            else:
                averaged_img.append(blur_img[i - 1][j - 1])
    averaged_img = np.reshape(averaged_img, (n_row, n_col))
    return averaged_img
    
    
def box_blur(image):
    """
    This function will calculate the blurred image.
    """
    square, square_row, blur_row, blur_img = [], [], [], []

    # rp is row pointer and cp is column pointer
    rp, cp = 0, 0 
      
    while rp <= len(image) - 3: 
        while cp <= len(image[0]) - 3:
              
            for i in range(rp, rp + 3):
                  
                for j in range(cp, cp + 3):
                      
                    square_row.append(image[i][j])
                      
                square.append(square_row)
                square_row = []
              
            blur_row.append(square_matrix(square))
            square = []             
            cp = cp + 1
          
        # append the blur_row in blur_image
        blur_img.append(blur_row)
        blur_row = []
        rp = rp + 1 # increase row pointer
        cp = 0 # start column pointer from 0 again
        
    # Adjust padding around edges and reshape to original shape
    new_img = concat_full_matrix(image, blur_img)
    # Return the resulting pixel matrix
    return new_img
 

def make_weights(in_tensor):
    """
    A function that returns the appropriate weight vector to statistically
    weight imbalanced data.
    """

    weight = torch.histc(in_tensor, bins=256, min = -0.5, max = 0.5)
    weight = 1/(weight+1e-3)
    
    return weight


def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """
    This function will load the most current training checkpoint.
    """
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


def find_latest_checkpoint(model_dir):
    """
    This helper function finds the checkpoint with the largest
    epoch value given a model directory.
    """
    tmp_dir = os.path.join(model_dir, 'checkpoints')
    list_of_files = glob.glob(tmp_dir+'/model_epoch_*.pth') 
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file