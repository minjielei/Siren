import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


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

def plot_losses(total_steps, train_losses, filename):
    x_steps = np.linspace(0, total_steps, num=total_steps)
    plt.figure(tight_layout=True)
    plt.plot(x_steps, train_losses)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_pred_vs_gt(pred, gt, filename):
    plt.figure(tight_layout=True)
    plt.scatter(gt*16, pred*16)
    plt.grid()        
    plt.xlabel('Truth')
    plt.ylabel('Pred')

    min_scale = min(plt.gca().get_xlim()[0], plt.gca().get_ylim()[0])
    max_scale = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])
    plt.xlim(min_scale, max_scale)
    plt.ylim(min_scale, max_scale)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_hist_overlap(pred, gt, filename):
    calc = (gt - pred)/(2*(gt + pred))
    plt.hist(calc.flatten(), alpha=0.5)
    plt.xlabel('asymmetry')
    plt.ylabel('samples')
    plt.yscale('log')
    plt.savefig(filename)
    plt.clf()

def draw_img(predict_video, ground_truth_video, model_dir):
    diff = np.sum(abs(ground_truth_video - predict_video), axis=(1,2,3))
    ground_truth_video = np.uint8((ground_truth_video * 1.0 + 0.5) * 255)
    predict_video = np.uint8((predict_video * 1.0 + 0.5) * 255)
    render_video = np.concatenate((ground_truth_video, predict_video), axis=1)

    im_name = os.path.join(model_dir, 'img.png')
    gt_name = os.path.join(model_dir, 'gt_img.png')
    pred_name = os.path.join(model_dir, 'pred_img.png')

    im_render = Image.fromarray(np.squeeze(render_video, -1) , 'L').convert('RGB')
    gt_im = Image.fromarray(np.squeeze(ground_truth_video,-1), 'L').convert('RGB')
    pred_im = Image.fromarray(np.squeeze(predict_video,-1), 'L').convert('RGB')

    gt_draw = ImageDraw.Draw(gt_im)
    pred_draw = ImageDraw.Draw(pred_im)
    draw = ImageDraw.Draw(im_render)
    draw.text((0, 0), "{:.2f}".format(diff), (255,0,0))

    im_render.save(im_name)
    gt_im.save(gt_name)
    pred_im.save(pred_name)

def get_mgrid(data_shape):
    x = np.linspace(0, 1, data_shape[0])
    y = np.linspace(0, 1, data_shape[1])
    z = np.linspace(0, 1, data_shape[2])
    
    coordx, coordy, coordz = np.meshgrid(x, y, z)
    coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))

    return coord
                            
class DataWrapper(DataLoader):
    def __init__(self, coord, data):
        # self.dataset = dataset
        self.mgrid = coord
        self.gt = data
        
    def __len__(self):
        return self.mgrid.shape[0]
    
    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.mgrid[idx]}
        gt_dict = {'idx': idx, 'coords': self.gt[idx]}

        return in_dict, gt_dict
    

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

    weight = torch.histc(in_tensor, bins=12, min = 0.0, max = 1.0)
    weight = weight.max()/weight
    
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
        total_steps = checkpoint['step']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {}, steps {})"
                  .format(filename, checkpoint['epoch'], checkpoint['step']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, total_steps, start_epoch, losslogger


def find_latest_checkpoint(model_dir):
    """
    This helper function finds the checkpoint with the largest
    epoch value given a model directory.
    """
    tmp_dir = os.path.join(model_dir, 'checkpoints')
    list_of_files = glob.glob(tmp_dir+'/model_epoch_*.pth') 
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file