import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
from datetime import datetime


from src.models.full_atom.module import FullAtomLitModule
from src.data.full_atom.datamodule import FullAtomDataModule
from src.models.full_atom.score_network import BaseScoreNetwork
from src.utils.get_args import get_args_Pre_cond

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


def train(data_cfg, model_cfg, train_cfg):

    data_module = FullAtomDataModule(data_cfg=data_cfg)
    model = FullAtomLitModule(model_cfg=model_cfg, train_cfg=train_cfg)

    logger = TensorBoardLogger(
        save_dir="./log/", name=train_cfg['log_path']
    )
    torch.set_float32_matmul_precision(precision="high")

    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",  # 要监控的指标
        dirpath=train_cfg['save_ckpt_path'],  # 保存 checkpoint 的目录
        # checkpoint 文件名格式
        filename="best-checkpoint-{epoch:02d}-{loss/val_loss:.4f}",
        save_top_k=-1,  # 只保存最好的 checkpoint
        mode="min",  # 根据监控指标的最低值保存
    )

    if train_cfg["strategy"] == "fsdp":
        # policy = {model.score_network.model_nn.structure_module,model.score_network.model_nn.embedder,}
        policy = {
            BaseScoreNetwork,
        }
        # activation_checkpointing_policy = {
        #     model.score_network.model_nn.structure_module,
        # }
        train_cfg["strategy"] = FSDPStrategy(
            auto_wrap_policy=policy,
            sharding_strategy="FULL_SHARD",
            # activation_checkpointing_policy=activation_checkpointing_policy,
            cpu_offload=True,
        )


    if train_cfg.get("Pretrain_ckpt_path"):
        # 加载预训练权重
        checkpoint = torch.load(train_cfg.get("Pretrain_ckpt_path"))

        # 提取模型的state_dict
        model_dict = model.state_dict()

        # 只加载匹配的层
        pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict and model_dict[k].shape == v.shape}

        # 更新模型的state_dict
        model_dict.update(pretrained_dict)

        # 重新加载模型
        model.load_state_dict(model_dict)


    
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
        use_distributed_sampler = False,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=model_cfg.get("ckpt_path") or None,
    )



if __name__ == "__main__":

    data_cfg, model_cfg, train_cfg = get_args_Pre_cond()
    
    
    print("当前日期和时间:", datetime.now())
    # 输出config文件
    print_dict_tree(
        {
            "data_cfg": data_cfg,
            "model_cfg": model_cfg,
            "train_cfg": train_cfg,
        }
    )

    train(data_cfg, model_cfg, train_cfg)
