import sys
import os
import time
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import loss_functions, modules, training, utils
from torch.utils.data import DataLoader

from functools import partial
import argparse
import torch
import torch.nn as nn
import numpy as np

from photon_library import PhotonLibrary


# Example syntax:
# run siren_test.py --output_dir result_72221 --batch_size 1 --experiment_name full_detector

# Configure Arguments
p = argparse.ArgumentParser()

p.add_argument('--output_dir', type=str, default='./results', help='root for logging outputs')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=100)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5') #5e-6 for FH
p.add_argument('--num_epochs', type=int, default=2000,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

opt = p.parse_args()

output_dir = os.path.join(opt.output_dir, opt.experiment_name)
weight_path = os.path.join(opt.output_dir, '_weights.pth')

device = list(range(torch.cuda.device_count()))

start = time.time()

# Load plib dataset
print('Load data ...')
plib = PhotonLibrary()
coord, data = plib.load_data()
coord = 2 * (coord - 0.5) 

start2 = time.time()

data = -np.log(data+1e-7)
data = (data - np.amin(data)) / (np.amax(data) - np.amin(data))

print('about to call cuda for first time...')
data = torch.from_numpy(data.astype(np.float32)).cuda()
print('Cuda finished')
data.requires_grad = False

print('Assigning Model...')
model = modules.Siren(in_features=3, out_features=180, hidden_features=512, hidden_layers=5, outermost_linear=True, omega=30)
model = model.float()
model = nn.DataParallel(model, device_ids=device)
model.cuda()

train_data = utils.DataWrapper(coord, data)

# Make weights
weight = utils.make_weights(data.flatten(), 12)
weight.cuda()
print('at the dataloader')

dataloader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size, pin_memory=False, num_workers=0)

loss = partial(loss_functions.image_mse)

print('Training...')
training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=output_dir, loss_fn=loss, k1=opt.kl_weight, weight = weight)

end = time.time()
print('Training Time: {}'.format(end-start2))
print('Complete. :)')
    
end = time.time()
print('Total Time: {}'.format(end-start))
