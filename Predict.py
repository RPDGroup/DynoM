import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy


from src.models.full_atom.module import FullAtomLitModule
from src.data.full_atom.datamodule import FullAtomDataModule
from src.models.full_atom.score_network import BaseScoreNetwork
import warnings

warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")


# 设置随机种子
L.seed_everything(42)


def print_dict_tree(obj, indent: str = "", is_last: bool = True) -> None:
    """按照树形结构打印配置文件"""
    branch = "└── " if is_last else "├── "
    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            last = i == len(obj) - 1
            print(f"{indent}{branch}{k}:", end="")
            # 在冒号后同一行直接打印基础值；若为容器则换行递归
            if isinstance(v, (dict, list)):
                print()
                print_dict_tree(v, indent + ("    " if is_last else "│   "), last)
            else:
                print(f" {v}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            last = i == len(obj) - 1
            print(f"{indent}{branch}[{i}]")
            print_dict_tree(item, indent + ("    " if is_last else "│   "), last)
    else:
        print(f"{indent}{branch}{obj}")


def eval(data_cfg, model_cfg, train_cfg):

    assert model_cfg["ckpt_path"] is not None, "ckpt_path must be specified"

    data_module = FullAtomDataModule(data_cfg=data_cfg)
    model = FullAtomLitModule(
        model_cfg=model_cfg, train_cfg=train_cfg, output_dir=train_cfg["output_dir"]
    )

    logger = TensorBoardLogger(save_dir="./log/", name="Pretrain_log")
    torch.set_float32_matmul_precision(precision="high")

    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",  # 要监控的指标
        dirpath="./Pretrain",  # 保存 checkpoint 的目录
        # checkpoint 文件名格式
        filename="best-checkpoint-{epoch:02d}-{loss/val_loss:.4f}",
        save_top_k=-1,  # 只保存最好的 checkpoint
        mode="min",  # 根据监控指标的最低值保存
    )

    if train_cfg["strategy"] == "fsdp":
        policy = {
            BaseScoreNetwork,
        }
        activation_checkpointing_policy = {
            model.score_network.model_nn.structure_module,
        }
        train_cfg["strategy"] = FSDPStrategy(
            auto_wrap_policy=policy,
            sharding_strategy="FULL_SHARD",
            activation_checkpointing_policy=activation_checkpointing_policy,
            cpu_offload=True,
        )

    trainer = L.Trainer(
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        strategy=train_cfg["strategy"],
        max_epochs=train_cfg["max_epochs"],
        logger=logger,
        precision=train_cfg["precision"],
        log_every_n_steps=train_cfg["log_every_n_steps"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        gradient_clip_val=train_cfg["gradient_clip_val"],
        gradient_clip_algorithm=train_cfg["gradient_clip_algorithm"],
        deterministic=train_cfg["deterministic"],
        inference_mode=train_cfg["inference_mode"],
        callbacks=[checkpoint_callback],
    )

    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=model_cfg.get("ckpt_path") or None,
    )



def get_argrs():
    parser = argparse.ArgumentParser(
        description="Configuration arguments for the model"
    )

    # data_cfg arguments
    parser.add_argument(
        "--gen_dataset_test_gen_dataset",
        type=str,
        required=True,
        help="Test dataset csv",
    )
    parser.add_argument(
        "--alphafold3_cfg_repr_data_root",
        type=str,
        required=True,
        help="alphafold3 representation data directory root",
    )
    parser.add_argument(
        "--alphafold3_cfg_seqres_to_index_path",
        type=str,
        required=True,
        help="Seqres to index path",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        required=True,
        help="Batch size for generation dataset",
    )
    parser.add_argument(
        "--gen_num_samples",
        type=int,
        default=1000,
        help="num samples for generation dataset",
    )

    # model_cfg arguments
    parser.add_argument(
        "--model_ckpt_path", type=str, required=True, help="Checkpoint path"
    )

    # train_cfg arguments
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for training"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_argrs()

    ## config
    data_cfg = {
        "mode": "val",
        "gen_dataset": {
            "test_gen_dataset": args.gen_dataset_test_gen_dataset,
            "gen_batch_size": args.gen_batch_size,
            "num_samples": args.gen_num_samples,
        },
        "is_clustering_training": False,
        "num_workers": 4,
        "train_batch_size": 5,
        "valid_batch_size": 5,
        "pin_memory": False,
        "use_alphafold3_repr": True,
        "alphafold3_cfg": {
            "complex_repr_data_root": args.alphafold3_cfg_repr_data_root,
            "monomer_repr_data_root": args.alphafold3_cfg_repr_data_root,
            "num_recycle": 10,
            "node_size": 384,
            "edge_size": 128,
            "seqres_to_index_path": args.alphafold3_cfg_seqres_to_index_path,
        },
        "se3_cfg": {
            "diffuse_trans": True,
            "diffuse_rot": True,
            "r3": {"min_b": 0.1, "max_b": 20.0, "coordinate_scaling": 0.1},
            "so3": {
                "num_omega": 1000,
                "num_sigma": 1000,
                "min_sigma": 0.1,
                "max_sigma": 1.5,
                "schedule": "logarithmic",
                "cache_dir": ".cache/",
                "use_cached_score": False,
            },
        },
    }

    model_cfg = {
        "model_name": "full_atom",
        "ckpt_path": args.model_ckpt_path,
        "model_nn": {
            "embedder": {
                "time_emb_size": 64,
                "scale_t": 1000.0,
                "res_idx_emb_size": 64,
                "r_max": 32,
                "num_rbf": 64,
                "rbf_min": 0.0,
                "rbf_max": 5.0,
                "pretrained_node_repr_size": (
                    384 if data_cfg["use_alphafold3_repr"] else 0
                ),  # alphafold3: 384, ESMFold: 1024
                "pretrained_edge_repr_size": (
                    128 if data_cfg["use_alphafold3_repr"] else 0
                ),
                "node_emb_size": 256,
                "edge_emb_size": 128,
                "use_af3_relative_pos_encoding": False,
                "dp_repr_size": 0,
                "use_dp_repr": False,
            },
            "structure_module": {
                "num_ipa_blocks": 4,
                "c_s": 256,
                "c_z": 128,
                "c_hidden": 256,
                "c_skip": 64,
                "no_heads": 4,
                "no_qk_points": 8,
                "no_v_points": 12,
                "seq_tfmr_num_heads": 4,
                "seq_tfmr_num_layers": 2,
            },
            "confidence_head": None,
        },
        "se3_cfg": {
            "diffuse_trans": True,
            "diffuse_rot": True,
            "r3": {"min_b": 0.1, "max_b": 20.0, "coordinate_scaling": 0.1},
            "so3": {
                "num_omega": 1000,
                "num_sigma": 1000,
                "min_sigma": 0.1,
                "max_sigma": 1.5,
                "schedule": "logarithmic",
                "cache_dir": ".cache/",
                "use_cached_score": False,
            },
        },
        "loss": {
            "rot_loss_weight": 0.5,
            "rot_angle_loss_t_filter": 0.2,
            "trans_loss_weight": 1.0,
            "bb_coords_loss_weight": 0.25,
            "bb_coords_loss_t_filter": 0.25,
            "bb_dist_map_loss_weight": 0.25,
            "bb_dist_map_loss_t_filter": 0.25,
            "torsion_loss_weight": 0.25,
            "fape_loss_weight": 1.0,
            "confidence_loss": {
                "pae_loss_cfg": {
                    "min_bin": 0,
                    "max_bin": 32,
                    "no_bins": 64,
                    "eps": 1e-6,
                },
                "pde_loss_cfg": {
                    "min_bin": 0,
                    "max_bin": 32,
                    "no_bins": 64,
                    "eps": 1e-6,
                },
                "plddt_loss_cfg": {
                    "min_bin": 0,
                    "max_bin": 1,
                    "no_bins": 50,
                    "is_nucleotide_threshold": 30.0,
                    "is_not_nucleotide_threshold": 15.0,
                    "eps": 1e-6,
                    "normalize": True,
                    "reduction": "mean",
                },
            },
        },
        "reverse_sample_cfg": {
            "num_samples": 10,
            "scale_coords": 0.1,
            "diffusion_steps": 200,
            "is_show_diffusing": False,
            "temperature": 1.0,
        },
    }

    train_cfg = {
        "optimizer": {
            "_target_": torch.optim.AdamW,
            "_partial_": True,
            "lr": 3e-4,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "_target_": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "_partial_": True,
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "threshold": 0.001,
            "min_lr": 1e-6,
        },
        "max_epochs": 300,
        "devices": -1,
        "strategy": "ddp",
        "precision": 32,
        "log_every_n_steps": 10,
        "lr_warmup_steps": 5000,
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "val_gen_every_n_epochs": 1000,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "deterministic": False,
        "inference_mode": False,
        "output_dir": args.output_dir,
    }

    eval(data_cfg=data_cfg,
         model_cfg=model_cfg,
         train_cfg=train_cfg)
