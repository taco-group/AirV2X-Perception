
# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch, resource
torch.multiprocessing.set_sharing_strategy('file_system')
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
from torch.cuda import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils

def train_parser():
    """
    Configure command line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="OpenCOOD Training")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                      help="Path to training configuration yaml file")
    parser.add_argument("--model_dir", default="",
                      help="Path to continue training from a checkpoint")
    parser.add_argument("--dist_url", default="env://",
                      help="URL used to set up distributed training")
    parser.add_argument("--fusion_method", "-f", default="intermediate",
                      help="Fusion method to use during inference")
    parser.add_argument("--rank", default=0, type=int,
                      help="Node rank for distributed training")
    parser.add_argument("--tag", default="default",
                      help="Tag for the training session")
    parser.add_argument("--worker", default=8, type=int,
                      help="Number of workers for data loading")
    parser.add_argument("--amp", action="store_true",
                      help="Enable automatic mixed precision training")
    return parser.parse_args()

def setup_dataloader(dataset, hypes, opt, is_train=True):
    """
    Set up data loader for training or validation.
    
    Args:
        dataset: Dataset instance
        hypes (dict): Configuration parameters
        opt: Command line arguments
        is_train (bool): Whether this is for training or validation
        
    Returns:
        DataLoader: Configured data loader
    """
    if opt.distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        if is_train:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, hypes["train_params"]["batch_size"], drop_last=True)
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=opt.worker,
                collate_fn=dataset.collate_batch_train,
                timeout=5200,
                pin_memory=True,
                prefetch_factor=1
            )
        else:
            loader = DataLoader(
                dataset,
                sampler=sampler,
                num_workers=opt.worker,
                collate_fn=dataset.collate_batch_train,
                timeout=5200,
                pin_memory=True,
                prefetch_factor=1
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=opt.worker,
            collate_fn=dataset.collate_batch_train,
            shuffle=is_train,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4
        )
    return loader

def validate_model(model, val_loader, criterion, epoch,
                   device, hypes, scaler=None):
    model.eval()
    total_loss, n_sample = 0.0, 0
    with torch.no_grad():
        for _, batch_data in tqdm(enumerate(val_loader),
                                  total=len(val_loader),
                                  desc="Validation", leave=False):
            if batch_data is None:
                continue

            if "scope" in hypes["name"] or "how2comm" in hypes["name"]:
                _batch_data = train_utils.to_device(batch_data[0], device)
                batch_data  = train_utils.to_device(batch_data,   device)
                with amp.autocast(enabled=scaler is not None):
                    out = model(batch_data)
                    loss = criterion(out, _batch_data["ego"]["label_dict"])
            else:
                batch_data = train_utils.to_device(batch_data, device)
                batch_data["ego"]["epoch"] = epoch
                with amp.autocast(enabled=scaler is not None):
                    out  = model(batch_data["ego"])
                    loss = criterion(out, batch_data["ego"]["label_dict"])

            bsz = 1 if isinstance(loss, torch.Tensor) else len(loss)
            total_loss += loss.item() * bsz   
            n_sample   += bsz

    return total_loss, n_sample

def main():
    """Main training function."""
    # Setup
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    hypes["tag"] = opt.tag
    
    # Build datasets
    print("Building datasets...")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    
    # Create dataloaders
    train_loader = setup_dataloader(train_dataset, hypes, opt, is_train=True)
    val_loader = setup_dataloader(val_dataset, hypes, opt, is_train=False)
    
    # Create model
    print("Creating model...")
    model = train_utils.create_model(hypes)
    total_params = sum(p.nelement() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup device and distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.to(device)
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True)
    
    # Setup training components
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    
    # Initialize mixed precision training if enabled
    scaler = amp.GradScaler() if opt.amp and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training")
    
    # Load checkpoint if continuing training
    if opt.model_dir:
        init_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, init_epoch=init_epoch, n_iter_per_epoch=len(train_loader))
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        print(f"Results will be saved to: {saved_path}")
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, n_iter_per_epoch=len(train_loader))
    
    # Training loop
    writer = SummaryWriter(saved_path)
    print("Starting training...")
    epochs = hypes["train_params"]["epoches"]
    
    for epoch in range(init_epoch, max(epochs, init_epoch)):
        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr}")
        
        # Training epoch
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch_data in pbar:
            if batch_data is None:
                continue
                
            # Forward pass
            model.zero_grad()
            optimizer.zero_grad()
            
            if "scope" in hypes["name"] or "how2comm" in hypes["name"]:
                _batch_data = batch_data[0]
                batch_data = train_utils.to_device(batch_data, device)
                _batch_data = train_utils.to_device(_batch_data, device)
                
                with amp.autocast(enabled=scaler is not None):
                    output_dict = model(batch_data)
                    loss = criterion(output_dict, _batch_data["ego"]["label_dict"])
            else:
                batch_data = train_utils.to_device(batch_data, device)
                batch_data["ego"]["epoch"] = epoch
                
                with amp.autocast(enabled=scaler is not None):
                    output_dict = model(batch_data["ego"])
                    loss = criterion(output_dict, batch_data["ego"]["label_dict"])
            
            # Backward pass with mixed precision support
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update progress bar
            try:
                print_msg = criterion.logging(epoch, i, len(train_loader), writer, pbar)
            except:
                print_msg = criterion.logging(epoch, i, len(train_loader), writer)
            if print_msg:
                pbar.set_description(print_msg)
            
            # Log training loss
            if opt.rank == 0:
                with open(os.path.join(saved_path, "train_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], iter[{i}/{len(train_loader)}], loss[{loss.item():.4f}]\n")
                    
        if opt.distributed:
            torch.cuda.synchronize()             
            torch.distributed.barrier(device_ids=[opt.gpu])  
        
        # Save checkpoint
        if opt.rank == 0 and epoch % hypes["train_params"]["save_freq"] == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(save_dict, os.path.join(saved_path, f"net_epoch{epoch + 1}.pth"))
        
        # Validation
        need_val = (epoch % hypes["train_params"]["eval_freq"] == 0)

        if need_val:
            if opt.distributed and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            local_sum, local_cnt = validate_model(model, val_loader,
                                                criterion, epoch,
                                                device, hypes, scaler)

            if opt.distributed:
                stats = torch.tensor([local_sum, local_cnt],
                                    dtype=torch.float32, device=device)
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                global_sum, global_cnt = stats.tolist()
            else:
                global_sum, global_cnt = local_sum, local_cnt

            if opt.rank == 0:
                val_loss = global_sum / max(global_cnt, 1)
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
                writer.add_scalar("Validate_Loss", val_loss, epoch)
                with open(os.path.join(saved_path, "validation_loss.txt"), "a+") as f:
                    f.write(f"Epoch[{epoch}], loss[{val_loss:.4f}]\n")

        
        scheduler.step(epoch)
                
        if opt.distributed:
            torch.distributed.barrier(device_ids=[opt.gpu])
        
    
    print(f"Training finished. Checkpoints saved to {saved_path}")
    torch.cuda.empty_cache()
    
    # Run inference after training
    fusion_method = opt.fusion_method
    cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
    print(f"Running inference: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
