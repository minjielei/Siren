import os
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
       (i.e. blurred pixel value) for each 3 * 3 blur image.
    """
    tot_sum = 0

    # Calculate sum of all the pixels in 3 * 3 matrix
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                tot_sum += square[i][j]
        # mean average of the neighboring pixels and the pixel value in question
    pixel_val = ((tot_sum // 8) + square[1][1]) // 2

    return pixel_val  # return the average of the sum of pixels and the central pixel value.


def concat_full_matrix(original, blur_img):
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
    This function will calculate the blurred 
    image
    """
    square = []     # This will store the 3 * 3 matrix 
                 # which will be used to find its blurred pixel
                   
    square_row = [] # This will store one row of a 3 * 3 matrix and 
                    # will be appended in square
                      
    blur_row = []   # Here we will store the resulting blurred
                    # pixels possible in one row 
                    # and will append this in the blur_img
      
    blur_img = [] # This is the resulting blurred image
      
    # number of rows in the given image
    n_row = len(image) 
      
    # number of columns in the given image
    n_col = len(image[0]) 
      
    # rp is row pointer and cp is column pointer
    rp, cp = 0, 0 
      
    # This while loop will be used to 
    # calculate all the blurred pixel in the first row 
    while rp <= n_row - 3: 
        while cp <= n_col-3:
              
            for i in range(rp, rp + 3):
                  
                for j in range(cp, cp + 3):
                      
                    # append all the pixels in a row of 3 * 3 matrix
                    square_row.append(image[i][j])
                      
                # append the row in the square i.e. 3 * 3 matrix 
                square.append(square_row)
                square_row = []
              
            # calculate the blurred pixel for given 3 * 3 matrix 
            # i.e. square and append it in blur_row
            blur_row.append(square_matrix(square))
            square = []
              
            # increase the column pointer
            cp = cp + 1
          
        # append the blur_row in blur_image
        blur_img.append(blur_row)
        blur_row = []
        rp = rp + 1 # increase row pointer
        cp = 0 # start column pointer from 0 again
        
    # Now adjust padding around edges and reshape to original shape
    new_img = concat_full_matrix(image, blur_img)
    # Return the resulting pixel matrix
    return new_img
 

def make_weights(in_tensor):

    weight = torch.histc(in_tensor, bins=256, min = -0.5, max = 0.5)
    weight = 1/(weight+1e-3)
    
    return weight