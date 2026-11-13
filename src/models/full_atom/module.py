"""Lightning training hooks, ensemble sampling, and PDB output for DynoM."""

from pathlib import Path
from typing import Dict, Any

import torch
from lightning import Fabric, LightningModule, seed_everything
from torchmetrics import MeanMetric, MinMetric
from src.openfold_local.np.protein import Protein
from src.openfold_local.np import protein
import os
import numpy as np
import torch.distributed as dist

from .score_network import BaseScoreNetwork



class FullAtomLitModule(LightningModule):
    """Wrap the score network with optimization, logging, and structure export.

    Training and validation call the supervised score-network forward pass.
    The test loop runs reverse_sample and writes ensembles beneath test_gen.
    An optional second validation loader generates structures periodically.
    """

    def __init__(
            self,
            model_cfg,
            train_cfg,
            output_dir: str ='./output',
            log_loss_name=[
                "total",
                "rot",
                "trans",
                "bb_coords",
                "bb_dist_map",
                "torsion",
                "fape",
                "clash",
            ],
            **kwargs,
    ):
        super().__init__()

        self.output_dir = output_dir
        self.val_output_dir = f'{output_dir}/val_gen/'
        self.test_output_dir = f'{output_dir}/test_gen/'
        

        self.score_network = BaseScoreNetwork(model_nn_cfg=model_cfg['model_nn'],
                                                loss_cfg=model_cfg['loss'],
                                                reverse_sample_cfg=model_cfg['reverse_sample_cfg'],
                                                se3_cfg=model_cfg['se3_cfg']
                                                )        
        self.train_cfg = train_cfg
        self.loss_cfg = model_cfg['loss']
        self.loss_name = log_loss_name

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                setattr(self, f"{split}_{loss_name}", MeanMetric())
        self.best_val_total = MinMetric()


    def setup(self, stage: str) -> None:
        # broadcast output_dir from rank 0
        fabric = Fabric()
        fabric.launch()
        self.output_dir = fabric.broadcast(self.output_dir, src=0)
        self.val_output_dir = fabric.broadcast(self.val_output_dir, src=0)
        self.test_output_dir = fabric.broadcast(self.test_output_dir, src=0)
               

    def forward(
            self,
            batch: Dict[str, Any]
    ):
        return self.score_network(**batch)

    def process_batch_step(self,batch, output, output_dir,model_out=None, step=None):
        """Remove batch padding and write final or intermediate atom37 coordinates.

        Internal chain IDs start at 1; Protein expects zero-based chain indices.
        Output residues are numbered consecutively across the concatenated chains.
        B-factors are written as zeros, not predicted confidence values.
        """

        for i in range(batch["aatype"].shape[0]):
            padding_mask = batch["padding_mask"][i]
            aatype = batch["aatype"][i][padding_mask].cpu().numpy()
            atom37 = output["atom37"][i][padding_mask].cpu().numpy() if output else output["atom37"][i][padding_mask].cpu().numpy()
            atom37_mask = output["atom37_mask"][i][padding_mask].cpu().numpy() if output else output["atom37_mask"][i][padding_mask].cpu().numpy()
            chain_ids = batch["chain_ids"][i] - 1
            chain_ids = chain_ids[padding_mask].cpu().numpy()
            res_idx = np.arange(aatype.shape[0])
            
            output_name = batch["output_name"][i]
            fname = batch["fname"][i]
            folder_path = f"{output_dir}/{fname}/"
            os.makedirs(folder_path, exist_ok=True)
                
            gen_protein = Protein(
                aatype=aatype,
                atom_positions=atom37,
                atom_mask=atom37_mask,
                residue_index=res_idx + 1,
                chain_index=chain_ids,
                b_factors=np.zeros_like(atom37_mask),
            )
            
            if step is not None:
                Path(f"{folder_path}/{Path(output_name).stem}").mkdir(parents=True, exist_ok=True)        
                with open(f"{folder_path}/{Path(output_name).stem}/{Path(output_name).stem}_{step}.pdb", "w") as fp:
                    fp.write(protein.to_pdb(gen_protein))
            else:
                with open(f"{folder_path}/{Path(output_name).stem}.pdb", "w") as fp:
                    fp.write(protein.to_pdb(gen_protein))
                    
    def sampling(self, batch, output_dir):
        """Run reverse diffusion and export final structures plus requested snapshots."""


        output, step_diffusion = self.score_network.reverse_sample(**batch)
        print(step_diffusion)
        # Export intermediate clean-structure predictions when snapshots are enabled.
        if len(step_diffusion) != 0:
            for idx,step in enumerate(step_diffusion):
                self.process_batch_step(batch, step, output_dir, step=step['step_idx'])

        # Export the final sampled structure for each item in the batch.
        self.process_batch_step(batch, output,output_dir)


    def on_train_start(self):
        """Called at the beginning of training after sanity check.

        NOTE: by default lightning executes validation step sanity checks before training starts,
              so it's worth to make sure validation metrics don't store results from these checks
        """
        local_rank = int(dist.get_rank())
        seed_everything(42 + local_rank, workers=True)

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                getattr(self, f"{split}_{loss_name}").reset()

        self.best_val_total.reset()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        
        try:    
            loss, aux_info = self.forward(batch)
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, sync_dist=True)
            
            for loss_name in self.loss_name:
                getattr(self, f"train_{loss_name}").update(aux_info[loss_name])
            
            self.log(
                "loss/train_loss",
                loss.item(),
                batch_size=batch["aatype"].shape[0],
                on_step = True,
                on_epoch = True,
                sync_dist = True,
                prog_bar = True,
            )
        except Exception as e:
            print(e)
            print(batch['fname'])
            raise e
        
        return loss
    
    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ):
        try:    
            # batch = self.move_to_cuda(batch)
            # The first loader measures supervised denoising loss; any later
            # loader is reserved for periodic ensemble generation.
            if dataloader_idx == 0:
                loss, aux_info = self.forward(batch)
                for loss_name in self.loss_name:
                    getattr(self, f"val_{loss_name}").update(aux_info[loss_name])
                
                self.log(
                    "loss/val_loss",
                    loss.item(),
                    batch_size=batch["aatype"].shape[0],
                    on_step = False,
                    on_epoch = True,
                    sync_dist = True,
                    prog_bar = True,
                )
                return loss

            elif (not self.trainer.sanity_checking) and (
                    (self.current_epoch + 1) %  self.train_cfg['val_gen_every_n_epochs'] == 0  # type: ignore
            ):
                # inference on val_gen dataset
                output_dir = f"{self.val_output_dir}/epoch{self.current_epoch}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                self.sampling(batch, output_dir)
        except Exception as e:
            print(e)
            print(batch['fname'])
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, sync_dist=True)
        log_info = f"Current epoch: {self.current_epoch:d}, step: {self.global_step:d}, lr: {lr:.8f}, "

        for split in ["train", "val"]:
            for loss_name in self.loss_name:
                epoch_loss = getattr(self, f"{split}_{loss_name}").compute()
                self.log(f"{split}/{loss_name}_loss", epoch_loss, sync_dist=True)
                log_info += f"{split}/{loss_name}_loss: {epoch_loss:.8f}, "
                getattr(self, f"{split}_{loss_name}").reset()
                if split == "val" and loss_name == "total":
                    self.best_val_total.update(epoch_loss)
                    self.log(
                        "val/best_val_total_loss",
                        self.best_val_total.compute(),
                        sync_dist=True,
                    )

        dist.barrier()
        # evaluate val_gen results
        output_dir = f"{self.val_output_dir}/epoch{self.current_epoch}"
        if self.trainer.is_global_zero and os.path.exists(output_dir):
            # log_stats, log_dist = eval_gen_conf(
            #     output_root=output_dir,
            #     csv_fpath=self.trainer.datamodule.val_gen_dataset.csv_path,
            #     ref_root=self.trainer.datamodule.val_gen_dataset.data_dir,
            #     num_samples=self.trainer.datamodule.val_gen_dataset.num_samples,
            #     n_proc=1,
            # )
            # log_stats = {
            #     f"val_gen/cameo/{name}": val for name, val in log_stats.items()
            # }
            # self.log_dict(log_stats, rank_zero_only=True, sync_dist=True)
            print("val_gen results are logged in the log file")

        torch.cuda.empty_cache()

    def on_test_start(self):
        local_rank = int(dist.get_rank())
        seed_everything(42 + local_rank, workers=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        # inference on test_gen dataset
        self.sampling(batch, self.test_output_dir)

    def on_test_end(self):
        dist.barrier()

    @property
    def is_epoch_based(self):
        """If the training is epoch-based or iteration-based."""
        return isinstance(self.trainer.val_check_interval, float) and self.trainer.val_check_interval <= 1.0

    def configure_optimizers(self):
        """Build the configured optimizer and optional validation-driven scheduler.

        The scheduler monitors val/total_loss at the configured validation cadence;
        optimizer_step applies a separate linear learning-rate warmup.
        """
        if self.train_cfg['optimizer']['_target_'] == "DeepSpeedCPUAdam":
            optimizer = self.train_cfg['optimizer']['_target_'](
                model_params=self.trainer.model.parameters(),
                lr=self.train_cfg['optimizer']['lr'],
                weight_decay=self.train_cfg['optimizer']['weight_decay']
            )
        else:
            optimizer = self.train_cfg['optimizer']['_target_'](params=self.trainer.model.parameters(),
                                                                lr=self.train_cfg['optimizer']['lr'],
                                                                weight_decay=self.train_cfg['optimizer']['weight_decay'])

        if self.train_cfg['scheduler'] is not None:
            scheduler = self.train_cfg['scheduler']['_target_'](optimizer=optimizer,
                                                                mode=self.train_cfg['scheduler']['mode'],
                                                                factor=self.train_cfg['scheduler']['factor'],
                                                                patience=self.train_cfg['scheduler']['patience'],
                                                                threshold=self.train_cfg['scheduler']['threshold'],
                                                                min_lr=self.train_cfg['scheduler']['min_lr'],)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'monitor': 'val/total_loss',
                    'interval': 'epoch' if self.is_epoch_based else 'step',
                    'strict': True,
                    'frequency': self.trainer.check_val_every_n_epoch if self.is_epoch_based else int(
                        self.trainer.val_check_interval),  # adjust lr_scheduler everytime run evaluation
                },
            }
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):

        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        # Warm up by optimizer steps, so gradient accumulation affects the number
        # of batches consumed before the configured base learning rate is reached.
        if self.trainer.global_step < self.train_cfg["lr_warmup_steps"]:  # type: ignore
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.train_cfg["lr_warmup_steps"]  # type: ignore
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.train_cfg['optimizer']["lr"]
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
