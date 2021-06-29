'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os
import shutil
import loss_functions, utils
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont



def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, data_shape, loss_fn, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    train_losses = []
    for epoch in range(epochs):
        if not epoch % epochs_til_checkpoint and epoch:
            print('epoch:', epoch )
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                       np.array(train_losses))

        for step, (model_input, gt) in enumerate(train_dataloader):
            start_time = time.time()
            
            
            model_input = {key: value.cuda() for key, value in model_input.items()}

            model_output = model(model_input['coords'])
            
            if not epoch%1000:
                model_output['model_out'] = utils.attach_tensor(utils.detach_tensor(model_output['model_out'], data_shape))

            losses = loss_functions.image_mse_TV_prior(1e-1, model, model_output, gt)

    
            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()

                if loss_schedules is not None and loss_name in loss_schedules:
                    writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                    single_loss *= loss_schedules[loss_name](total_steps)

                writer.add_scalar(loss_name, single_loss, total_steps)
                train_loss += single_loss

            train_losses.append(train_loss.item())
            writer.add_scalar("total_train_loss", train_loss, total_steps)

            if not total_steps % steps_til_summary:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_current.pth'))
#                 summary_fn(model, model_input, gt, model_output, writer, total_steps)

            optim.zero_grad()
            train_loss.backward()
            optim.step()

            # if not total_steps % steps_til_summary:
            #
            #     if val_dataloader is not None:
            #         print("Running validation set...")
            #         model.eval()
            #         with torch.no_grad():
            #             val_losses = []
            #             for (model_input, gt) in val_dataloader:
            #                 model_output = model(model_input)
            #                 val_loss = loss_fn(model_output, gt)
            #                 val_losses.append(val_loss)
            #
            #             writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
            #         model.train()

            total_steps += 1

    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
               np.array(train_losses))
    
    #Plot and save loss
    x_steps = np.linspace(0, total_steps, num=total_steps)
    plt.figure(tight_layout=True)
    plt.plot(x_steps, train_losses)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt_name = os.path.join(model_dir, 'total_loss.png')
    plt.savefig(plt_name, dpi=300, bbox_inches='tight')
    plt.clf()
    
    
    # Make images
    ground_truth_video = np.reshape(
          gt['coords'].cpu().detach().numpy(), 
          (data_shape[0], data_shape[1], data_shape[2], -1)
        )
    predict_video = np.reshape(
            model_output['model_out'].cpu().detach().numpy(), 
            (data_shape[0], data_shape[1], data_shape[2], -1)
        )

    diff = np.sum(abs(ground_truth_video - predict_video), axis=(1,2,3))
    ground_truth_video = np.uint8((ground_truth_video * 1.0 + 0.5) * 255)
    predict_video = np.uint8((predict_video * 1.0 + 0.5) * 255)
    render_video = np.concatenate((ground_truth_video, predict_video), axis=1)

    for step in range(predict_video.shape[0]):
        im_name = os.path.join(model_dir, '{:05d}.png'.format(step))
        gt_name = os.path.join(model_dir, '{:05d}_gt.png'.format(step))
        pred_name = os.path.join(model_dir, '{:05d}_pred.png'.format(step))


        im_render = Image.fromarray(np.squeeze(render_video[step], -1) , 'L').convert('RGB')
        gt_im = Image.fromarray(np.squeeze(ground_truth_video[step],-1), 'L').convert('RGB')
        pred_im = Image.fromarray(np.squeeze(predict_video[step],-1), 'L').convert('RGB')

        
        gt_draw = ImageDraw.Draw(gt_im)
        pred_draw = ImageDraw.Draw(pred_im)
        draw = ImageDraw.Draw(im_render)
        draw.text((0, 0), "{:.2f}".format(diff[step]), (255,0,0))

        im_render.save(im_name)
        gt_im.save(gt_name)
        pred_im.save(pred_name)



class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

    