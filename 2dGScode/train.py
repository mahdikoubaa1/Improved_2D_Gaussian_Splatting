#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim,ScaleAndShiftInvariantLoss, get_normal_loss
from lpipsPyTorch import lpips
import json
from utils.mono_prior import estimate_depth, estimate_normal, get_decay_factor
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from utils.point_utils import depth_to_normal, depths_to_points
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
from torch.utils.tensorboard import SummaryWriter
import numpy as np




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    # MCMC requires cap_max to be set if enabled
    if opt.mcmc and dataset.cap_max == -1:
        print("MCMC enabled: Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        gaussians.training_setup(opt)
    depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1, reduction='image-based')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mono_depth_for_log = 0.0
    ema_mono_normal_l1_for_log = 0.0
    ema_mono_normal_cos_for_log = 0.0
    ema_opacity_reg_for_log = 0.0
    ema_scale_reg_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)
        

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # Monocular prior losses (Task 4) - MonoSDF style with exponential decay
        mono_depth_loss = torch.tensor(0.0, device="cuda")
        mono_normal_l1_loss = torch.tensor(0.0, device="cuda")
        mono_normal_cos_loss = torch.tensor(0.0, device="cuda")
        
        # Compute exponential decay for monocular priors
        mono_depth_prior_start_iter = 4000
        depth_decay = get_decay_factor(opt, iteration, mono_depth_prior_start_iter)
        if opt.lambda_mono_depth > 0 and iteration > mono_depth_prior_start_iter:
            surf_depth = render_pkg['surf_depth']
            surf_depth = 1/(surf_depth+ 1e-8)
            surf_depth = (surf_depth - surf_depth.min()) / (surf_depth.max() - surf_depth.min()+ 1e-8)
            with torch.no_grad():
                if viewpoint_cam.mono_depth is None:
                    depth_estimate = (estimate_depth(viewpoint_cam) ) 
                    viewpoint_cam.mono_depth = depth_estimate.cpu() if depth_estimate is not None else None
                else:
                    depth_estimate = viewpoint_cam.mono_depth.cuda()
            mono_depth_loss =  opt.lambda_mono_depth * depth_loss(
                surf_depth, depth_estimate, mask= torch.ones_like(surf_depth, device="cuda")
            )
        
        mono_normal_prior_start_iter = 8000
        normal_decay = get_decay_factor(opt, iteration, mono_normal_prior_start_iter)
        if (opt.lambda_mono_normal_l1 > 0 or opt.lambda_mono_normal_cos > 0) and iteration > mono_normal_prior_start_iter:
            # Returns separate L1 and cosine losses
            with torch.no_grad():
                if viewpoint_cam.mono_normal is None:
                    normal_estimate = estimate_normal(viewpoint_cam)
                    viewpoint_cam.mono_normal = normal_estimate.cpu() if normal_estimate is not None else None
                    
                else:
                    normal_estimate = viewpoint_cam.mono_normal.cuda()
        
            normal_l1, normal_cos = get_normal_loss(rend_normal.permute(1, 2, 0).cuda(), normal_estimate.permute(1, 2, 0).cuda())
            mono_normal_l1_loss = opt.lambda_mono_normal_l1 * normal_l1
            mono_normal_cos_loss = opt.lambda_mono_normal_cos * normal_cos

        # MCMC regularization losses
        opacity_reg_loss = 0.0
        scale_reg_loss = 0.0
        if opt.mcmc:
            opacity_reg_loss = opt.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            scale_reg_loss = opt.scale_reg * torch.abs(gaussians.get_scaling).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + depth_decay * mono_depth_loss + normal_decay * mono_normal_l1_loss + normal_decay * mono_normal_cos_loss + opacity_reg_loss + scale_reg_loss  
        
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_mono_depth_for_log = 0.4 * mono_depth_loss.item() + 0.6 * ema_mono_depth_for_log
            ema_mono_normal_l1_for_log = 0.4 * mono_normal_l1_loss.item() + 0.6 * ema_mono_normal_l1_for_log
            ema_mono_normal_cos_for_log = 0.4 * mono_normal_cos_loss.item() + 0.6 * ema_mono_normal_cos_for_log
            ema_opacity_reg_for_log = 0.4 * opacity_reg_loss + 0.6 * ema_opacity_reg_for_log
            ema_scale_reg_for_log = 0.4 * scale_reg_loss + 0.6 * ema_scale_reg_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "mono_depth": f"{ema_mono_depth_for_log:.{5}f}",
                    "mono_normal_l1": f"{ema_mono_normal_l1_for_log:.{5}f}",
                    "mono_normal_cos": f"{ema_mono_normal_cos_for_log:.{5}f}",
                    "opacity_reg": f"{ema_opacity_reg_for_log:.{5}f}",
                    "scale_reg": f"{ema_scale_reg_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                # MonoSDF-style monocular prior losses
                if mono_depth_loss.item() > 0:
                    tb_writer.add_scalar('train_loss_patches/mono_depth_loss', mono_depth_loss.item(), iteration)
                if mono_normal_l1_loss.item() > 0:
                    tb_writer.add_scalar('train_loss_patches/mono_normal_l1_loss', mono_normal_l1_loss.item(), iteration)
                if mono_normal_cos_loss.item() > 0:
                    tb_writer.add_scalar('train_loss_patches/mono_normal_cos_loss', mono_normal_cos_loss.item(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),save=iteration in saving_iterations, model_path=scene.model_path)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                # MCMC Densification
                if opt.mcmc:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                        gaussians.relocate_gs(dead_mask=dead_mask)
                        gaussians.add_new_gs(cap_max=dataset.cap_max)
                # Original Heuristic Densification
                else:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Depth Reinitialization (Task 2) - Mini-Splatting strategy
                if opt.depth_reinit_every > 0 and iteration % opt.depth_reinit_every == 0:
                    out_pts_list=[]
                    gt_list=[]
                    views=scene.getTrainCameras()
                    for view in tqdm(views):
                        gt = view.original_image[0:3, :, :]
                        render_pkg = render(view, gaussians, pipe, background)
                        out_pts = depths_to_points(view, render_pkg["surf_depth"])
                        accum_alpha = render_pkg["rend_alpha"]


                        prob = accum_alpha

                        prob = prob/prob.sum()
                        prob = prob.reshape(-1).cpu().numpy()


                        factor=1/(gt.shape[1]*gt.shape[2]*len(views)/(opt.reinit_target_points))


                        N_xyz=prob.shape[0]
                        num_sampled=int(N_xyz*factor)
                        print (gt.shape)
                        indices = np.random.choice(N_xyz, size=num_sampled, 
                                                   p=prob,replace=False)
                        gt = gt.permute(1,2,0).reshape(-1,3)

                        out_pts_list.append(out_pts[indices])
                        gt_list.append(gt[indices])       

    

                    out_pts_merged=torch.cat(out_pts_list)
                    gt_merged=torch.cat(gt_list)

                    gaussians.reinitial_pts(out_pts_merged, gt_merged)
                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                    viewpoint_stack = scene.getTrainCameras().copy()
            # Final reinitialization at densify_until_iter (Mini-Splatting strategy)
            # This resets Gaussian parameters for the final training phase
            #if iteration == opt.densify_until_iter and len(opt.depth_reinit_iters) > 0:
            #    from utils.sh_utils import SH2RGB
            #    print(f"\n[ITER {iteration}] Final Reinitialization at densify_until_iter (Mini-Splatting)")
            #    
            #    # Reinit with current positions but reset other parameters
            #    # This is exactly what Mini-Splatting does at this point
            #    current_xyz = gaussians._xyz.detach().clone()
            #    current_rgb = SH2RGB(gaussians._features_dc.detach().squeeze(1))
            #    
            #    gaussians.reinitialize_from_depth(current_xyz, current_rgb, opt)
            #    torch.cuda.empty_cache()
            #    viewpoint_stack = None


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                # SGLD noise injection for MCMC exploration
                if opt.mcmc:
                    L = build_scaling_rotation(
                        torch.cat([gaussians.get_scaling, torch.zeros_like(gaussians.get_scaling[:, :1])], dim=-1),
                        gaussians.get_rotation
                    )
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * op_sigmoid(1 - gaussians.get_opacity) * opt.noise_lr * xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, save=False, model_path=''):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None],   global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, 'vgg').mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                if save and config['name'] == 'test':
                    metrics = {}
                    metrics['L1'] = l1_test.item()
                    metrics['PSNR'] = psnr_test.item()
                    metrics['LPIPS'] = lpips_test.item()
                    metrics['SSIM'] = ssim_test.item()
                    metrics['Points'] = scene.gaussians.get_xyz.shape[0]
                    results_dir = os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration))
                    os.makedirs(results_dir, exist_ok=True)
                    with open(os.path.join(results_dir, f'metrics.json'), 'w') as f:
                        json.dump(metrics, f)
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters",conflict_handler='resolve')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(15000, 60001, 15000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i for i in range(30000, 60001, 1000)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
