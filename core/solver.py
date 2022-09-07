"""
Note: CIT-GAN is built over StarGAN v2 to improve the performance
Author of CIT-GAN: Shivangi Yadav
Advisor: Dr. Arun Ross

Reference:
StarGAN v2: https://github.com/clovaai/stargan-v2/
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
import numpy as np

from time import sleep
from tqdm import tqdm

import pdb
from pathlib import Path


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train' or args.mode == 'test_detector':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, "{:06d}_nets.ckpt"), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, "{:06d}_nets_ema.ckpt"), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, "{:06d}_optims.ckpt"), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(Path(ospj(args.checkpoint_dir, "{:06d}_nets_ema.ckpt")), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        #pdb.set_trace()
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
        #pdb.set_trace()
        #
        # Uncomment this when training CIT-GAN for first time to load pre-trained Styling Network
        #
        #if torch.cuda.is_available():
        #    module_dict = torch.load('D:\\cit-gan\\codes\\expr\\checkpoints-style2\\000017_nets_prestyle.ckpt')
        #else:
        #    module_dict = torch.load('D:\\cit-gan\\codes\\expr\\checkpoints-style2\\000017_nets_prestyle.ckpt', map_location=torch.device('cpu'))
        #
        #for name, module in module_dict.items():
        #    if 1:
        #        self.nets.style_encoder.module.load_state_dict(module_dict[name])
        #        self.nets_ema.style_encoder.module.load_state_dict(module_dict[name])
        #    else:
        #        self.nets.style_encoder.load_state_dict(module_dict[name])
        #        self.nets_ema.style_encoder.load_state_dict(module_dict[name])
        
        

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        
        #pdb.set_trace()

        for i in tqdm(range(args.resume_iter, args.total_iters)):
            # fetch images and labels
            sleep(3)
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref

            #Instead of starting from randomized latent vector (as done in Stargan v2), CIT-GAN utilizes
            #Style Vector to train generator
            # Old
            # z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
            # New
            #pdb.set_trace()
            z_trg, _ = nets.style_encoder(x_ref, y_trg)
            z_trg2, _ = nets.style_encoder(x_ref2, y_trg)


            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            print("Training Discriminator")
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()


            # train Style Network
            print("Training Style Network")
            for sty_iter in range(0, 10):
                #print("Training Sty.... {}", sty_iter)
                sty_loss = compute_s_loss(
                    nets, args, x_real, y_org)
                self._reset_grad()
                sty_loss.backward()
                optims.style_encoder.step()


            # train the generator
            #for i in range(0, args.iter_gen):
            print("Training Generator")
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            #optims.mapping_network.step()
            #optims.style_encoder.step()


            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)
            #moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')
    
    def test_detector(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        
        # resume training if necessary
        
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
        
        
        #if torch.cuda.is_available():
        #    module_dict = torch.load('D:\\cit-gan\\codes\\expr\\checkpoints-style2\\000017_nets_prestyle.ckpt')
        #else:
        #    module_dict = torch.load('D:\\cit-gan\\codes\\expr\\checkpoints-style2\\000017_nets_prestyle.ckpt', map_location=torch.device('cpu'))

        #for name, module in module_dict.items():
        #    if 1:
        #        self.nets.style_encoder.module.load_state_dict(module_dict[name])
        #        self.nets_ema.style_encoder.module.load_state_dict(module_dict[name])
        #    else:
        #        self.nets.style_encoder.load_state_dict(module_dict[name])
        #        self.nets_ema.style_encoder.load_state_dict(module_dict[name])

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start testing...')
        start_time = time.time()
        total_samples = len(loaders.val.dataset)
        nets.style_encoder.eval()
        
        with torch.no_grad():
            running_loss = 0.0
            running_corrects = 0.0
            
            for j, inputs in tqdm(enumerate(loaders.val)):
                pdb.set_trace()
                sleep(3)
                x_real, y_org, path = inputs
                x_real = x_real.cuda()
                y_org = y_org.cuda()
                
                s_real, o_real = nets.style_encoder(x_real, y_org)
                #pdb.set_trace()
                _, preds = torch.max(o_real, 1)
                
                
                # print out log info
                running_corrects += torch.sum(preds == y_org.data)
                #print("Running Loss: ", loss.item())
                #print("Running Corrects: ", torch.sum(preds == y_org.data))
                
        epoch_acc = running_corrects.double() / total_samples
        print("Accuracy: {}".format(epoch_acc))

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    _, out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    #pdb.set_trace()

    # with fake images
    with torch.no_grad():

        if z_trg is not None:
            s_trg = z_trg
        else:  # x_ref is not None
            s_trg, _ = nets.style_encoder(x_ref, y_trg)


        x_fake = nets.generator(x_real, s_trg, masks=masks)
    _, out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, cls_loss = nn.CrossEntropyLoss()):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = z_trg
    else:
        s_trg, h_trg = nets.style_encoder(x_ref, y_trg)
    
    _, h_real = nets.style_encoder(x_real, y_org)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    _, out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # updated style reconstruction loss
    s_pred, h_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    if z_trgs is not None:
        s_trg2 = z_trg2
    else:
        s_trg2, h_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_cls = torch.mean(torch.abs(x_fake - x_fake2)) + 0.1 * cls_loss(h_real, y_trg)

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org, h_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_cls + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_cls.item(),
                       cyc=loss_cyc.item())


def compute_s_loss(nets, args, x_real, y_org, cls_loss = nn.CrossEntropyLoss()):
    #pdb.set_trace()
    x_real.requires_grad_()
    _, o_real = nets.style_encoder(x_real, y_org)
    #pdb.set_trace()
    _, preds = torch.max(o_real, 1)
    loss = cls_loss(o_real, y_org)
    return loss

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
