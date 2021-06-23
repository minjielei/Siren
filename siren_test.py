# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import loss_functions, modules, training, utils
from torch.utils.data import DataLoader

from functools import partial
import configargparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photon_library import PhotonLibrary

# Example syntax:
# run siren_test.py --max_x 3 --output_dir result_61621 --experiment_name first_test

# Configure Arguments
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--output_dir', type=str, default='./results', help='root for logging outputs')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--min_x', type=int, default=2,
                    help='Minimum number of slices in x axis')
p.add_argument('--max_x', type=int, default=3,
                    help='Maximum number of slices in x axis')
p.add_argument('--skip_x', type=int, default=2,
                    help='Number of slices skipped in x axis')
p.add_argument('--batch_size', type=int, default=100)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5') #5e-6 for FH
p.add_argument('--num_epochs', type=int, default=20000,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

opt = p.parse_args()


# Load plib dataset
print('Load data ...')
plib = PhotonLibrary()
full_data = plib.numpy() # data_shape

opt.min_x = min(opt.min_x, opt.max_x - 1)
for s in range(opt.max_x, opt.min_x, -1 * opt.skip_x):
    print('Starting to slice...')

    output_dir = os.path.join(opt.output_dir, opt.experiment_name)
    weight_path = os.path.join(opt.output_dir, '_weights.pth')

    data = full_data[:s, :, :, :]
    data_shape = data.shape[0:-1]

    data = np.expand_dims(np.reshape(data, (-1, data.shape[-1])), axis=0)
    data = np.expand_dims(np.sum(data, -1), axis=-1)
    data = (data - np.amin(data)) / (np.amax(data) - np.amin(data)) - 0.5
    print('about to call cuda')
    data = torch.from_numpy(data.astype(np.float32)).cuda()
    print('Cuda finished')
    data.requires_grad = False
    data = {'coords': data}
    
    x = np.linspace(0, data_shape[0] - 1, data_shape[0])
    y = np.linspace(0, data_shape[1] - 1, data_shape[1])
    z = np.linspace(0, data_shape[2] - 1, data_shape[2])
    
    print('organizing coords')
    coordx, coordy, coordz = np.meshgrid(x, y, z)
    coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))
    coord_real = np.expand_dims(plib._min + (plib._max - plib._min) / plib.shape * (coord + 0.5), axis=0)
    coord_real = torch.from_numpy(coord_real.astype(np.float32)).cuda()

    coord_real.requires_grad = False
    
    coord_real = {'coords': coord_real}
    
    print('Assigning Model...')
    model = modules.Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=1, outermost_linear=True)
    model = model.float()
    model.cuda()
    print('at the dataloader')
    model_output= model(coord_real['coords'])

    train_data = utils.DataWrapper(model_output, data_shape, data)
    
    dataloader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size, pin_memory=False, num_workers=0)
    
    print('At loss')
    loss = loss_functions.image_mse_TV_prior(opt.kl_weight, model, model_output, data)

    print('Training...')
    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=output_dir, data_shape=data_shape, loss_fn=loss)
    
    print('Complete. :)')

