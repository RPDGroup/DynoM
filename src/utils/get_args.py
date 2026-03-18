import argparse
import torch
from lightning.pytorch.strategies import DDPStrategy



def get_args():
    """
    Parses command-line arguments for the Unconditional Protein Diffusion Model Pre-training.
    Returns a dictionary containing data_cfg, model_cfg, and train_cfg.
    """
    parser = argparse.ArgumentParser(description="Configuration for Unconditional Protein Diffusion Model Pre-training")

    # Data Configuration Group
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--data_target_dataset", type=str, default="RCSBDataset", help="Target dataset class.")
    data_group.add_argument("--data_train_csv_path", type=str,required=True,help="Path to the training CSV metadata file.")
    data_group.add_argument("--data_val_csv_path", type=str,
                            required=True,
                            help="Path to the validation CSV metadata file.")
    data_group.add_argument("--data_mode", type=str, default="train", help="Dataset mode (e.g., 'train', 'val').")
    data_group.add_argument("--data_train_monomer_pdb_data_dir", type=str,
                            help="Directory train_monomer_pdb_data_dir training PDB data.")
    data_group.add_argument("--data_val_monomer_pdb_data_dir", type=str,
                            help="Directory val_monomer_pdb_data_dir validation PDB data.")
    data_group.add_argument("--data_train_complex_pdb_data_dir", type=str,
                            help="Directory train_complex_pdb_data_dir validation PDB data.")
    data_group.add_argument("--data_val_complex_pdb_data_dir", type=str,
                            help="Directory val_complex_pdb_data_dir validation PDB data.")
    data_group.add_argument("--data_is_clustering_training", type=bool, default=False,
                            help="Whether to use clustering for training.")
    data_group.add_argument("--data_train_num_workers", type=int, default=8,
                            help="Number of workers for training data loading.")
    data_group.add_argument("--data_val_num_workers", type=int, default=8,
                            help="Number of workers for validation data loading.")
    data_group.add_argument("--data_train_batch_size", type=int, default=2, help="Batch size for training.")
    data_group.add_argument("--data_valid_batch_size", type=int, default=4, help="Batch size for validation.")
    data_group.add_argument("--data_pin_memory", type=bool, default=True, help="Pin memory for data loading.")

    # DeepRank-GNN / dp_cfg
    data_group.add_argument("--data_dp_data_root", type=str, required=False,
                        help="Optional: DeepRank-GNN representation directory.")

    # Data CSV Processor Configuration (Train)
    data_group.add_argument("--data_train_csv_processor_min_seqlen", type=int, default=0,
                            help="Minimum sequence length for training CSV processing.")
    data_group.add_argument("--data_train_csv_processor_max_seqlen", type=int, default=100000,
                            help="Maximum sequence length for training CSV processing.")
    data_group.add_argument("--data_train_csv_processor_max_coil_ratio", type=float, default=0.5,
                            help="Maximum coil ratio for training CSV processing.")
    data_group.add_argument("--data_train_csv_processor_min_valid_frame_ratio", type=float, default=0.7,
                            help="Minimum valid frame ratio for training CSV processing.")
    data_group.add_argument("--data_train_csv_processor_groupby", type=str, default=None,
                            help="Groupby columns for training CSV processing.")

    # Data CSV Processor Configuration (Valid)
    data_group.add_argument("--data_valid_csv_processor_min_seqlen", type=int, default=0,
                            help="Minimum sequence length for validation CSV processing.")
    data_group.add_argument("--data_valid_csv_processor_max_seqlen", type=int, default=100000,
                            help="Maximum sequence length for validation CSV processing.")
    data_group.add_argument("--data_valid_csv_processor_max_coil_ratio", type=float, default=0.5,
                            help="Maximum coil ratio for validation CSV processing.")
    data_group.add_argument("--data_valid_csv_processor_min_valid_frame_ratio", type=float, default=0.7,
                            help="Minimum valid frame ratio for validation CSV processing.")

    # alphafold3 Configuration
    data_group.add_argument("--data_use_alphafold3_repr", type=bool, default=False,
                            help="Whether to use alphafold3 representations.")
    data_group.add_argument("--data_alphafold3_seqres_to_index_path", type=str, default=None)
    data_group.add_argument("--data_complex_repr_data_root", type=str,
                            help="Root directory for alphafold3 data.")
    data_group.add_argument("--data_monomer_repr_data_root", type=str,
                            help="Root directory for alphafold3 data.")
    data_group.add_argument("--data_alphafold3_num_recycle", type=int, default=10,
                            help="Number of recycles for alphafold3.")
    data_group.add_argument("--data_alphafold3_node_size", type=int, default=384, help="Node size for alphafold3 features.")
    data_group.add_argument("--data_alphafold3_edge_size", type=int, default=128, help="Edge size for alphafold3 features.")

    # SE3 Configuration (Data)
    data_group.add_argument("--data_se3_diffuse_trans", type=bool, default=True,
                            help="Enable translational diffusion in SE3 for data.")
    data_group.add_argument("--data_se3_diffuse_rot", type=bool, default=True,
                            help="Enable rotational diffusion in SE3 for data.")
    data_group.add_argument("--data_se3_r3_min_b", type=float, default=0.1,
                            help="Minimum 'b' for R3 diffusion in data.")
    data_group.add_argument("--data_se3_r3_max_b", type=float, default=20.0,
                            help="Maximum 'b' for R3 diffusion in data.")
    data_group.add_argument("--data_se3_r3_coordinate_scaling", type=float, default=0.1,
                            help="Coordinate scaling for R3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_num_omega", type=int, default=1000,
                            help="Number of omega steps for SO3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_num_sigma", type=int, default=1000,
                            help="Number of sigma steps for SO3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_min_sigma", type=float, default=0.1,
                            help="Minimum sigma for SO3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_max_sigma", type=float, default=1.5,
                            help="Maximum sigma for SO3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_schedule", type=str, default="logarithmic",
                            help="Schedule type for SO3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_cache_dir", type=str, default=".cache/",
                            help="Cache directory for SO3 diffusion in data.")
    data_group.add_argument("--data_se3_so3_use_cached_score", type=bool, default=False,
                            help="Use cached score for SO3 diffusion in data.")

    data_group.add_argument("--data_is_classify_sample", type=bool, default=False)
    data_group.add_argument("--data_is_order_sample", type=bool, default=False)
    data_group.add_argument("--data_classify_num", type=str, default=None)
    # Model Configuration Group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_name", type=str, default="unconditional_model_cfg",
                             help="Name of the model configuration.")
    model_group.add_argument("--model_ckpt_path", type=str, default=None,)
    
    # Model NN Embedder
    model_group.add_argument("--use_dp_repr", type=bool, default=False,
                             help="Use dp_repr or Not.")  # dp_repr_size
    model_group.add_argument("--model_nn_embedder_dp_repr_size", type=int, default=64,
                             help="Size of dp_repr embedding.")  # dp_repr_size
    model_group.add_argument("--model_nn_embedder_time_emb_size", type=int, default=64, help="Time embedding size.")
    model_group.add_argument("--model_nn_embedder_scale_t", type=float, default=1000.0,
                             help="Scaling factor for time embedding.")
    model_group.add_argument("--model_nn_embedder_res_idx_emb_size", type=int, default=64,
                             help="Residual index embedding size.")
    model_group.add_argument("--model_nn_embedder_num_rbf", type=int, default=64, help="Number of RBF functions.")
    model_group.add_argument("--model_nn_embedder_r_max", type=int, default=32,
                             help="Maximum radius for RBF embeddings.")
    model_group.add_argument("--model_nn_embedder_rbf_min", type=float, default=0.0, help="Minimum value for RBF.")
    model_group.add_argument("--model_nn_embedder_rbf_max", type=float, default=5.0, help="Maximum value for RBF.")
    

    # We'll set these dynamically in the reconstructed config.
    model_group.add_argument("--model_nn_embedder_node_emb_size", type=int, default=256, help="Node embedding size.")
    model_group.add_argument("--model_nn_embedder_edge_emb_size", type=int, default=128, help="Edge embedding size.")
    model_group.add_argument("--model_nn_embedder_use_af3_relative_pos_encoding", type=bool, default=False,
                             help="Use AlphaFold3 relative positional encoding.")

    # Model NN Structure Module
    model_group.add_argument("--model_nn_structure_module_num_ipa_blocks", type=int, default=4,
                             help="Number of IPA blocks in the structure module.")
    model_group.add_argument("--model_nn_structure_module_c_s", type=int, default=256, help="c_s for structure module.")
    model_group.add_argument("--model_nn_structure_module_c_z", type=int, default=128, help="c_z for structure module.")
    model_group.add_argument("--model_nn_structure_module_c_hidden", type=int, default=256,
                             help="c_hidden for structure module.")
    model_group.add_argument("--model_nn_structure_module_c_skip", type=int, default=64,
                             help="c_skip for structure module.")
    model_group.add_argument("--model_nn_structure_module_no_heads", type=int, default=4,
                             help="Number of heads in IPA attention.")
    model_group.add_argument("--model_nn_structure_module_no_qk_points", type=int, default=8,
                             help="Number of QK points in IPA.")
    model_group.add_argument("--model_nn_structure_module_no_v_points", type=int, default=12,
                             help="Number of V points in IPA.")
    model_group.add_argument("--model_nn_structure_module_seq_tfmr_num_heads", type=int, default=4,
                             help="Number of heads for sequence transformer.")
    model_group.add_argument("--model_nn_structure_module_seq_tfmr_num_layers", type=int, default=2,
                             help="Number of layers for sequence transformer.")

    # SE3 Configuration (Model) - Note: In your original config, this is identical to data_cfg's se3_cfg.
    # For argparse, they are distinct arguments.
    model_group.add_argument("--model_se3_diffuse_trans", type=bool, default=True,
                             help="Enable translational diffusion in SE3 for model.")
    model_group.add_argument("--model_se3_diffuse_rot", type=bool, default=True,
                             help="Enable rotational diffusion in SE3 for model.")
    model_group.add_argument("--model_se3_r3_min_b", type=float, default=0.1,
                             help="Minimum 'b' for R3 diffusion in model.")
    model_group.add_argument("--model_se3_r3_max_b", type=float, default=20.0,
                             help="Maximum 'b' for R3 diffusion in model.")
    model_group.add_argument("--model_se3_r3_coordinate_scaling", type=float, default=0.1,
                             help="Coordinate scaling for R3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_num_omega", type=int, default=1000,
                             help="Number of omega steps for SO3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_num_sigma", type=int, default=1000,
                             help="Number of sigma steps for SO3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_min_sigma", type=float, default=0.1,
                             help="Minimum sigma for SO3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_max_sigma", type=float, default=1.5,
                             help="Maximum sigma for SO3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_schedule", type=str, default="logarithmic",
                             help="Schedule type for SO3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_cache_dir", type=str, default=".cache/",
                             help="Cache directory for SO3 diffusion in model.")
    model_group.add_argument("--model_se3_so3_use_cached_score", type=bool, default=False,
                             help="Use cached score for SO3 diffusion in model.")

    # Model Loss Configuration
    model_group.add_argument("--model_loss_rot_loss_weight", type=float, default=0.5,
                             help="Weight for rotational loss.")
    model_group.add_argument("--model_loss_rot_angle_loss_t_filter", type=float, default=0.2,
                             help="Time filter for rotational angle loss.")
    model_group.add_argument("--model_loss_trans_loss_weight", type=float, default=1.0,
                             help="Weight for translational loss.")
    model_group.add_argument("--model_loss_bb_coords_loss_weight", type=float, default=0.25,
                             help="Weight for backbone coordinates loss.")
    model_group.add_argument("--model_loss_bb_coords_loss_t_filter", type=float, default=0.25,
                             help="Time filter for backbone coordinates loss.")
    model_group.add_argument("--model_loss_bb_dist_map_loss_weight", type=float, default=0.25,
                             help="Weight for backbone distance map loss.")
    model_group.add_argument("--model_loss_bb_dist_map_loss_t_filter", type=float, default=0.25,
                             help="Time filter for backbone distance map loss.")
    model_group.add_argument("--model_loss_torsion_loss_weight", type=float, default=0.25,
                             help="Weight for torsion loss.")
    model_group.add_argument("--model_loss_fape_loss_weight", type=float, default=1.0, help="Weight for FAPE loss.")
    model_group.add_argument("--model_loss_clash_loss_weight", type=float, default=0.3, help="Weight for clash loss.")
    model_group.add_argument("--model_loss_bb_dist_map_cutoff", type=float, default=8.5,
                             help="Cutoff for backbone distance map.")
    model_group.add_argument("--model_loss_violation_config_violation_tolerance_factor", type=float, default=12.0,
                             help="Violation tolerance factor for clash loss.")
    model_group.add_argument("--model_loss_violation_config_clash_overlap_tolerance", type=float, default=1.5,
                             help="Clash overlap tolerance for clash loss.")
    model_group.add_argument("--model_loss_violation_config_eps", type=float, default=1e-6,
                             help="Epsilon for violation config.")
    model_group.add_argument("--model_loss_violation_config_weight", type=float, default=0.0,
                             help="Weight for violation config.")

    # Model Reverse Sample Configuration
    model_group.add_argument("--model_reverse_sample_num_samples", type=int, default=10,
                             help="Number of samples for reverse diffusion.")
    model_group.add_argument("--model_reverse_sample_scale_coords", type=float, default=0.1,
                             help="Scale coordinates for reverse diffusion.")
    model_group.add_argument("--model_reverse_sample_diffusion_steps", type=int, default=1000,
                             help="Number of diffusion steps for reverse sampling.")
    model_group.add_argument("--model_reverse_sample_is_show_diffusing", type=bool, default=False,
                             help="Show diffusing process during reverse sampling.")

    # Train Configuration Group
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--train_optimizer_lr", type=float, default=3e-4, help="Learning rate for optimizer.")
    train_group.add_argument("--train_optimizer_weight_decay", type=float, default=0.0,
                             help="Weight decay for optimizer.")
    train_group.add_argument("--train_scheduler_mode", type=str, default="min", help="Mode for LR scheduler (min/max).")
    train_group.add_argument("--train_scheduler_factor", type=float, default=0.5, help="Factor by which to reduce LR.")
    train_group.add_argument("--train_scheduler_patience", type=int, default=10,
                             help="Patience (epochs) for LR scheduler.")
    train_group.add_argument("--train_scheduler_threshold", type=float, default=0.001,
                             help="Threshold for LR scheduler (metric change).")
    train_group.add_argument("--train_scheduler_min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    train_group.add_argument("--train_max_epochs", type=int, default=400, help="Maximum number of training epochs.")
    train_group.add_argument("--train_devices", type=int, default=-1,
                             help="Number of devices to use for training (-1 for all available).")
    train_group.add_argument("--train_precision", type=str, default="32",
                             help="Training precision (e.g., '32', '16-mixed').")
    train_group.add_argument("--train_log_every_n_steps", type=int, default=10, help="Log every n training steps.")
    train_group.add_argument("--train_lr_warmup_steps", type=int, default=5000,
                             help="Number of learning rate warmup steps.")
    train_group.add_argument("--train_accelerator", type=str,
                             default="cuda" if torch.cuda.is_available() else "cpu",
                             help="Accelerator to use for training ('cuda' or 'cpu').")
    train_group.add_argument("--train_val_gen_every_n_epochs", type=int, default=1000,
                             help="Generate validation samples every n epochs.")
    train_group.add_argument("--train_accumulate_grad_batches", type=int, default=4,
                             help="Number of batches to accumulate gradients over.")
    train_group.add_argument("--train_gradient_clip_val", type=float, default=1.0, help="Gradient clipping value.")
    train_group.add_argument("--train_gradient_clip_algorithm", type=str, default="norm",
                             help="Gradient clipping algorithm ('norm' or 'value').")
    train_group.add_argument("--train_deterministic", type=bool, default=False, help="Enable deterministic training.")
    train_group.add_argument("--train_inference_mode", type=bool, default=False, help="Enable inference mode.")
    train_group.add_argument("--train_Pretrain_ckpt_path", type=str, default=None, help="Random seed for training.")
    train_group.add_argument("--train_log_path", type=str, default="cond_monomer_Pretrain", help="Path for logging.")
    train_group.add_argument("--save_ckpt_path", type=str, default="./checkpoint/Pretrain",
                             help="Path for checkpoints.")
    args = parser.parse_args()

    return args


def get_args_Pre_uncond():

    args = get_args()
    
    # Reconstruct the nested dictionary structure from parsed arguments
    data_cfg = {
        "target_dataset": args.data_target_dataset,
        "train_csv_path": args.data_train_csv_path,
        "val_csv_path": args.data_val_csv_path,
        "mode": args.data_mode,
        "train_monomer_pdb_data_dir": args.data_train_monomer_pdb_data_dir,
        "val_monomer_pdb_data_dir": args.data_val_monomer_pdb_data_dir,
        "train_complex_pdb_data_dir": args.data_train_complex_pdb_data_dir,
        "val_complex_pdb_data_dir": args.data_val_complex_pdb_data_dir,
        "dataset": {
            "val_gen_dataset": None,
            "test_gen_dataset": None,
        },
        "is_clustering_training": args.data_is_clustering_training,
        "train_num_workers": args.data_train_num_workers,
        "val_num_workers": args.data_val_num_workers,
        "train_batch_size": args.data_train_batch_size,
        "valid_batch_size": args.data_valid_batch_size,
        "pin_memory": args.data_pin_memory,
        "train_csv_processor_cfg": {
            "min_seqlen": args.data_train_csv_processor_min_seqlen,
            "max_seqlen": args.data_train_csv_processor_max_seqlen,
            "max_coil_ratio": args.data_train_csv_processor_max_coil_ratio,
            "min_valid_frame_ratio": args.data_train_csv_processor_min_valid_frame_ratio,
            "groupby": [args.data_train_csv_processor_groupby] if args.data_train_csv_processor_groupby else None,
        },
        "valid_csv_processor_cfg": {
            "min_seqlen": args.data_valid_csv_processor_min_seqlen,
            "max_seqlen": args.data_valid_csv_processor_max_seqlen,
            "max_coil_ratio": args.data_valid_csv_processor_max_coil_ratio,
            "min_valid_frame_ratio": args.data_valid_csv_processor_min_valid_frame_ratio,
        },
        "use_alphafold3_repr": False, # uncond model
        "alphafold3_cfg": {
            "complex_repr_data_root": args.data_complex_repr_data_root,
            "monomer_repr_data_root":args.data_monomer_repr_data_root,
            'num_recycle': args.data_alphafold3_num_recycle,
            'node_size': args.data_alphafold3_node_size,
            'edge_size': args.data_alphafold3_edge_size,
        },
        "se3_cfg": {
            "diffuse_trans": args.data_se3_diffuse_trans,
            "diffuse_rot": args.data_se3_diffuse_rot,
            "r3": {
                "min_b": args.data_se3_r3_min_b,
                "max_b": args.data_se3_r3_max_b,
                "coordinate_scaling": args.data_se3_r3_coordinate_scaling
            },
            "so3": {
                "num_omega": args.data_se3_so3_num_omega,
                "num_sigma": args.data_se3_so3_num_sigma,
                "min_sigma": args.data_se3_so3_min_sigma,
                "max_sigma": args.data_se3_so3_max_sigma,
                "schedule": args.data_se3_so3_schedule,
                "cache_dir": args.data_se3_so3_cache_dir,
                "use_cached_score": args.data_se3_so3_use_cached_score
            }
        },
        "dp_cfg": (
            {"dp_root": args.data_dp_data_root}
            if args.data_dp_data_root is not None and args.data_dp_data_root.lower() != "none"
            else None
        )
    }

    model_cfg = {
        'model_name': args.model_name,
        "model_nn": {
            "embedder": {
                "time_emb_size": args.model_nn_embedder_time_emb_size,
                "scale_t": args.model_nn_embedder_scale_t,
                "res_idx_emb_size": args.model_nn_embedder_res_idx_emb_size,
                "num_rbf": args.model_nn_embedder_num_rbf,
                "r_max": args.model_nn_embedder_r_max,
                "rbf_min": args.model_nn_embedder_rbf_min,
                "rbf_max": args.model_nn_embedder_rbf_max,
                "pretrained_node_repr_size": 384 if args.data_use_alphafold3_repr else 0,
                "pretrained_edge_repr_size": 128 if args.data_use_alphafold3_repr else 0,
                "node_emb_size": args.model_nn_embedder_node_emb_size,
                "edge_emb_size": args.model_nn_embedder_edge_emb_size,
                "dp_repr_size": args.model_nn_embedder_dp_repr_size, 
                "use_dp_repr": args.use_dp_repr,
                "use_af3_relative_pos_encoding": args.model_nn_embedder_use_af3_relative_pos_encoding,
            },
            "structure_module": {
                "num_ipa_blocks": args.model_nn_structure_module_num_ipa_blocks,
                "c_s": args.model_nn_structure_module_c_s,
                "c_z": args.model_nn_structure_module_c_z,
                "c_hidden": args.model_nn_structure_module_c_hidden,
                "c_skip": args.model_nn_structure_module_c_skip,
                "no_heads": args.model_nn_structure_module_no_heads,
                "no_qk_points": args.model_nn_structure_module_no_qk_points,
                "no_v_points": args.model_nn_structure_module_no_v_points,
                "seq_tfmr_num_heads": args.model_nn_structure_module_seq_tfmr_num_heads,
                "seq_tfmr_num_layers": args.model_nn_structure_module_seq_tfmr_num_layers
            },
            "confidence_head": {
                "n_blocks": 0,
                "c_s": 384,
                "c_z": 128,
                "c_s_inputs": 384,
                "b_pae": 64,
                "b_pde": 64,
                "b_plddt": 50,
                "b_resolved": 2,
                "max_atoms_per_token": 20,
                "pairformer_dropout": 0,
                "blocks_per_ckpt": None,
                "distance_bin_start": 3.25,
                "distance_bin_end": 52,
                "distance_bin_step": 1.25,
                "stop_gradient": True
            },
        },
        "se3_cfg": {
            "diffuse_trans": args.model_se3_diffuse_trans,
            "diffuse_rot": args.model_se3_diffuse_rot,
            "r3": {
                "min_b": args.model_se3_r3_min_b,
                "max_b": args.model_se3_r3_max_b,
                "coordinate_scaling": args.model_se3_r3_coordinate_scaling
            },
            "so3": {
                "num_omega": args.model_se3_so3_num_omega,
                "num_sigma": args.model_se3_so3_num_sigma,
                "min_sigma": args.model_se3_so3_min_sigma,
                "max_sigma": args.model_se3_so3_max_sigma,
                "schedule": args.model_se3_so3_schedule,
                "cache_dir": args.model_se3_so3_cache_dir,
                "use_cached_score": args.model_se3_so3_use_cached_score
            }
        },
        "loss": {
            "rot_loss_weight": args.model_loss_rot_loss_weight,
            "rot_angle_loss_t_filter": args.model_loss_rot_angle_loss_t_filter,
            "trans_loss_weight": args.model_loss_trans_loss_weight,
            "bb_coords_loss_weight": args.model_loss_bb_coords_loss_weight,
            "bb_coords_loss_t_filter": args.model_loss_bb_coords_loss_t_filter,
            "bb_dist_map_loss_weight": args.model_loss_bb_dist_map_loss_weight,
            "bb_dist_map_loss_t_filter": args.model_loss_bb_dist_map_loss_t_filter,
            "torsion_loss_weight": args.model_loss_torsion_loss_weight,
            "fape_loss_weight": args.model_loss_fape_loss_weight,
            "clash_loss_weight": args.model_loss_clash_loss_weight,
            "bb_dist_map_cutoff": args.model_loss_bb_dist_map_cutoff,
            "pae_loss_weight" : 1,
            "pde_loss_weight" : 1,
            "plddt_loss_weight" : 1,
            "violation_config": {
                "violation_tolerance_factor": args.model_loss_violation_config_violation_tolerance_factor,
                "clash_overlap_tolerance": args.model_loss_violation_config_clash_overlap_tolerance,
                "eps": args.model_loss_violation_config_eps,
                "weight": args.model_loss_violation_config_weight,
            },
            "confidence_loss_cfg":{
                "pae_loss_cfg":{
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
            "num_samples": args.model_reverse_sample_num_samples,
            "scale_coords": args.model_reverse_sample_scale_coords,
            "diffusion_steps": args.model_reverse_sample_diffusion_steps,
            "is_show_diffusing": args.model_reverse_sample_is_show_diffusing,
        }
    }

    train_cfg = {
        "optimizer": {
            "_target_": torch.optim.AdamW,
            "_partial_": True,
            "lr": args.train_optimizer_lr,
            "weight_decay": args.train_optimizer_weight_decay
        },
        "scheduler": {
            "_target_": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "_partial_": True,
            "mode": args.train_scheduler_mode,
            "factor": args.train_scheduler_factor,
            "patience": args.train_scheduler_patience,
            "threshold": args.train_scheduler_threshold,
            "min_lr": args.train_scheduler_min_lr
        },
        'max_epochs': args.train_max_epochs,
        'devices': args.train_devices,
        'strategy': DDPStrategy(gradient_as_bucket_view=True),  # This is a direct object instantiation
        'precision': args.train_precision,
        "log_path": args.train_log_path,
        'log_every_n_steps': args.train_log_every_n_steps,
        "lr_warmup_steps": args.train_lr_warmup_steps,
        'accelerator': args.train_accelerator,
        "val_gen_every_n_epochs": args.train_val_gen_every_n_epochs,
        "accumulate_grad_batches": args.train_accumulate_grad_batches,
        'gradient_clip_val': args.train_gradient_clip_val,
        'gradient_clip_algorithm': args.train_gradient_clip_algorithm,
        'deterministic': args.train_deterministic,
        'inference_mode': args.train_inference_mode,
        "save_ckpt_path": args.save_ckpt_path,
        'Pretrain_ckpt_path': args.train_Pretrain_ckpt_path,
    }

    return data_cfg, model_cfg, train_cfg


def get_args_Pre_cond():

    args = get_args()

    config = {
        "data_cfg": {
            "target_dataset": args.data_target_dataset,
            "train_csv_path": args.data_train_csv_path,
            "val_csv_path": args.data_val_csv_path,
            "mode": args.data_mode,
            "train_monomer_pdb_data_dir": args.data_train_monomer_pdb_data_dir,
            "val_monomer_pdb_data_dir": args.data_val_monomer_pdb_data_dir,
            "train_complex_pdb_data_dir": args.data_train_complex_pdb_data_dir,
            "val_complex_pdb_data_dir": args.data_val_complex_pdb_data_dir,
            "dataset": {
                "val_gen_dataset": None,
                "test_gen_dataset": None,
            },
            "is_clustering_training": args.data_is_clustering_training,
            "train_num_workers": args.data_train_num_workers,
            "val_num_workers": args.data_val_num_workers,
            "train_batch_size": args.data_train_batch_size,
            "valid_batch_size": args.data_valid_batch_size,
            "pin_memory": args.data_pin_memory,
            "train_csv_processor_cfg": {
                "min_seqlen": args.data_train_csv_processor_min_seqlen,
                "max_seqlen": args.data_train_csv_processor_max_seqlen,
                "max_coil_ratio": args.data_train_csv_processor_max_coil_ratio,
                "min_valid_frame_ratio": args.data_train_csv_processor_min_valid_frame_ratio,
                "groupby": [args.data_train_csv_processor_groupby] if args.data_train_csv_processor_groupby else None
            },
            "valid_csv_processor_cfg": {
                "min_seqlen": args.data_valid_csv_processor_min_seqlen,
                "max_seqlen": args.data_valid_csv_processor_max_seqlen,
                "max_coil_ratio": args.data_valid_csv_processor_max_coil_ratio,
                "min_valid_frame_ratio": args.data_valid_csv_processor_min_valid_frame_ratio,
            },
            "use_alphafold3_repr": True, # cond model
            "alphafold3_cfg": {
                "complex_repr_data_root": args.data_complex_repr_data_root,
                "monomer_repr_data_root":args.data_monomer_repr_data_root,
                "num_recycle": args.data_alphafold3_num_recycle,
                "node_size": args.data_alphafold3_node_size,
                "edge_size": args.data_alphafold3_edge_size,
                "seqres_to_index_path": args.data_alphafold3_seqres_to_index_path
            },
            "se3_cfg": {
                "diffuse_trans": args.data_se3_diffuse_trans,
                "diffuse_rot": args.data_se3_diffuse_rot,
                "r3": {"min_b": args.data_se3_r3_min_b, "max_b": args.data_se3_r3_max_b,
                       "coordinate_scaling": args.data_se3_r3_coordinate_scaling},
                "so3": {
                    "num_omega": args.data_se3_so3_num_omega,
                    "num_sigma": args.data_se3_so3_num_sigma,
                    "min_sigma": args.data_se3_so3_min_sigma,
                    "max_sigma": args.data_se3_so3_max_sigma,
                    "schedule": args.data_se3_so3_schedule,
                    "cache_dir": args.data_se3_so3_cache_dir,
                    "use_cached_score": args.data_se3_so3_use_cached_score,
                },
            },
            "dp_cfg": (
                {"dp_root": args.data_dp_data_root}
                if args.data_dp_data_root is not None and args.data_dp_data_root.lower() != "none"
                else None
            )
        },
        "model_cfg": {
            "model_name": args.model_name,
            "ckpt_path": args.model_ckpt_path,
            "model_nn": {
                "embedder": {
                    "time_emb_size": args.model_nn_embedder_time_emb_size,
                    "scale_t": args.model_nn_embedder_scale_t,
                    "res_idx_emb_size": args.model_nn_embedder_res_idx_emb_size,
                    "r_max": args.model_nn_embedder_r_max,
                    "num_rbf": args.model_nn_embedder_num_rbf,
                    "rbf_min": args.model_nn_embedder_rbf_min,
                    "rbf_max": args.model_nn_embedder_rbf_max,
                    "pretrained_node_repr_size": (
                        384 if args.data_use_alphafold3_repr else 0
                    ),
                    "pretrained_edge_repr_size": (
                        128 if args.data_use_alphafold3_repr else 0
                    ),
                    "node_emb_size": args.model_nn_embedder_node_emb_size,
                    "edge_emb_size": args.model_nn_embedder_edge_emb_size,      
                    "dp_repr_size": args.model_nn_embedder_dp_repr_size,       
                    "use_dp_repr": args.use_dp_repr,       
                    "use_af3_relative_pos_encoding": args.model_nn_embedder_use_af3_relative_pos_encoding,
                },
                "structure_module": {
                    "num_ipa_blocks": args.model_nn_structure_module_num_ipa_blocks,
                    "c_s": args.model_nn_structure_module_c_s,
                    "c_z": args.model_nn_structure_module_c_z,
                    "c_hidden": args.model_nn_structure_module_c_hidden,
                    "c_skip": args.model_nn_structure_module_c_skip,
                    "no_heads": args.model_nn_structure_module_no_heads,
                    "no_qk_points": args.model_nn_structure_module_no_qk_points,
                    "no_v_points": args.model_nn_structure_module_no_v_points,
                    "seq_tfmr_num_heads": args.model_nn_structure_module_seq_tfmr_num_heads,
                    "seq_tfmr_num_layers": args.model_nn_structure_module_seq_tfmr_num_layers,
                },
                "confidence_head": None,
                # "confidence_head": {
                #     "n_blocks": 0,
                #     "c_s": 256,
                #     "c_z": 128,
                #     "c_s_inputs": 256,
                #     "b_pae": 64,
                #     "b_pde": 64,
                #     "b_plddt": 50,
                #     "b_resolved": 2,
                #     "max_atoms_per_token": 20,
                #     "pairformer_dropout": 0,
                #     "blocks_per_ckpt": None,
                #     "distance_bin_start": 3.25,
                #     "distance_bin_end": 52,
                #     "distance_bin_step": 1.25,
                #     "stop_gradient": True
                # },
            },
            "se3_cfg": {
                "diffuse_trans": args.model_se3_diffuse_trans,
                "diffuse_rot": args.model_se3_diffuse_rot,
                "r3": {"min_b": args.model_se3_r3_min_b, 
                       "max_b": args.model_se3_r3_max_b,
                       "coordinate_scaling": args.model_se3_r3_coordinate_scaling
                },
                "so3": {
                    "num_omega": args.model_se3_so3_num_omega,
                    "num_sigma": args.model_se3_so3_num_sigma,
                    "min_sigma": args.model_se3_so3_min_sigma,
                    "max_sigma": args.model_se3_so3_max_sigma,
                    "schedule": args.model_se3_so3_schedule,
                    "cache_dir": args.model_se3_so3_cache_dir,
                    "use_cached_score": args.model_se3_so3_use_cached_score,
                },
            },
            "loss": {
                "rot_loss_weight": args.model_loss_rot_loss_weight,
                "rot_angle_loss_t_filter": args.model_loss_rot_angle_loss_t_filter,
                "trans_loss_weight": args.model_loss_trans_loss_weight,
                "bb_coords_loss_weight": args.model_loss_bb_coords_loss_weight,
                "bb_coords_loss_t_filter": args.model_loss_bb_coords_loss_t_filter,
                "bb_dist_map_loss_weight": args.model_loss_bb_dist_map_loss_weight,
                "bb_dist_map_loss_t_filter": args.model_loss_bb_dist_map_loss_t_filter,
                "torsion_loss_weight": args.model_loss_torsion_loss_weight,
                "fape_loss_weight": args.model_loss_fape_loss_weight,
                "clash_loss_weight": args.model_loss_clash_loss_weight,
                "bb_dist_map_cutoff": args.model_loss_bb_dist_map_cutoff,
                "pae_loss_weight" : 1e-4,
                "pde_loss_weight" : 1e-4,
                "plddt_loss_weight" : 1e-4,
                "violation_config": {
                    "violation_tolerance_factor": args.model_loss_violation_config_violation_tolerance_factor,
                    "clash_overlap_tolerance": args.model_loss_violation_config_clash_overlap_tolerance,
                    "eps": args.model_loss_violation_config_eps,
                    "weight": args.model_loss_violation_config_weight,
                },
                "confidence_loss": None,
                # "confidence_loss":{
                #     "pae_loss_cfg":{
                #         "min_bin": 0,
                #         "max_bin": 32,
                #         "no_bins": 64,
                #         "eps": 1e-6,
                #     },
                #     "pde_loss_cfg": {
                #         "min_bin": 0,
                #         "max_bin": 32,
                #         "no_bins": 64,
                #         "eps": 1e-6,
                #     },
                #     "plddt_loss_cfg": {
                #         "min_bin": 0,
                #         "max_bin": 1,
                #         "no_bins": 50,
                #         "is_nucleotide_threshold": 30.0,
                #         "is_not_nucleotide_threshold": 15.0,
                #         "eps": 1e-6,
                #         "normalize": True,
                #         "reduction": "mean",
                #     },
                # },
            },
            "reverse_sample_cfg": {
                "num_samples": args.model_reverse_sample_num_samples,
                "scale_coords": args.model_reverse_sample_scale_coords,
                "diffusion_steps": args.model_reverse_sample_diffusion_steps,
                "is_show_diffusing": args.model_reverse_sample_is_show_diffusing,
            },
        },
        "train_cfg": {
            "optimizer": {
                "_target_": torch.optim.AdamW,
                "_partial_": True,
                "lr": args.train_optimizer_lr,
                "weight_decay": args.train_optimizer_weight_decay,
            },
            "scheduler": {
                "_target_": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "_partial_": True,
                "mode": args.train_scheduler_mode,
                "factor": args.train_scheduler_factor,
                "patience": args.train_scheduler_patience,
                "threshold": args.train_scheduler_threshold,
                "min_lr": args.train_scheduler_min_lr,
            },
            "max_epochs": args.train_max_epochs,
            "devices": args.train_devices,
            "strategy": DDPStrategy(gradient_as_bucket_view=True),
            "precision": args.train_precision,
            "log_every_n_steps": args.train_log_every_n_steps,
            "lr_warmup_steps": args.train_lr_warmup_steps,
            "accelerator": args.train_accelerator,
            "val_gen_every_n_epochs": args.train_val_gen_every_n_epochs,
            "accumulate_grad_batches": args.train_accumulate_grad_batches,
            "gradient_clip_val": args.train_gradient_clip_val,
            "gradient_clip_algorithm": args.train_gradient_clip_algorithm,
            "deterministic": args.train_deterministic,
            "inference_mode": args.train_inference_mode,
            "log_path": args.train_log_path,
            "save_ckpt_path": args.save_ckpt_path,
            'Pretrain_ckpt_path': args.train_Pretrain_ckpt_path,
        }
    }

    return config["data_cfg"], config["model_cfg"], config["train_cfg"]


def get_args_MD_finetune():
    
    args = get_args()

    # Reconstruct the nested dictionary structure from flat arguments
    config = {
        "data_cfg": {
            "target_dataset": args.data_target_dataset,
            "train_csv_path": args.data_train_csv_path,
            "val_csv_path": args.data_val_csv_path,
            "mode": args.data_mode,
            "train_monomer_pdb_data_dir": args.data_train_monomer_pdb_data_dir,
            "val_monomer_pdb_data_dir": args.data_val_monomer_pdb_data_dir,
            "train_complex_pdb_data_dir": args.data_train_complex_pdb_data_dir,
            "val_complex_pdb_data_dir": args.data_val_complex_pdb_data_dir,
            "dataset": {
                "val_gen_dataset": None,
                "test_gen_dataset": None,
            },
            "is_clustering_training": args.data_is_clustering_training,
            "train_num_workers": args.data_train_num_workers,
            "val_num_workers": args.data_val_num_workers,
            "train_batch_size": args.data_train_batch_size,
            "valid_batch_size": args.data_valid_batch_size,
            "pin_memory": args.data_pin_memory,
            "is_classify_sample": args.data_is_classify_sample,
            "is_order_sample":args.data_is_order_sample,
            "classify_num":args.data_classify_num,
            "train_csv_processor_cfg": {
                "min_seqlen": args.data_train_csv_processor_min_seqlen,
                "max_seqlen": args.data_train_csv_processor_max_seqlen,
                "max_coil_ratio": args.data_train_csv_processor_max_coil_ratio,
                "min_valid_frame_ratio": args.data_train_csv_processor_min_valid_frame_ratio,
            },
            "valid_csv_processor_cfg": {
                "min_seqlen": args.data_valid_csv_processor_min_seqlen,
                "max_seqlen": args.data_valid_csv_processor_max_seqlen,
                "max_coil_ratio": args.data_valid_csv_processor_max_coil_ratio,
                "min_valid_frame_ratio": args.data_valid_csv_processor_min_valid_frame_ratio,
                "num_samples": 1,
            },
            "use_alphafold3_repr": True, # cond
            "alphafold3_cfg": {
                "complex_repr_data_root": args.data_complex_repr_data_root,
                "monomer_repr_data_root":args.data_monomer_repr_data_root,
                "num_recycle": args.data_alphafold3_num_recycle,
                "node_size": args.data_alphafold3_node_size,
                "edge_size": args.data_alphafold3_edge_size,
                "seqres_to_index_path": args.data_alphafold3_seqres_to_index_path
            },
            "se3_cfg": {
                "diffuse_trans": args.data_se3_diffuse_trans,
                "diffuse_rot": args.data_se3_diffuse_rot,
                "r3": {"min_b": args.data_se3_r3_min_b, "max_b": args.data_se3_r3_max_b,
                       "coordinate_scaling": args.data_se3_r3_coordinate_scaling},
                "so3": {
                    "num_omega": args.data_se3_so3_num_omega,
                    "num_sigma": args.data_se3_so3_num_sigma,
                    "min_sigma": args.data_se3_so3_min_sigma,
                    "max_sigma": args.data_se3_so3_max_sigma,
                    "schedule": args.data_se3_so3_schedule,
                    "cache_dir": args.data_se3_so3_cache_dir,
                    "use_cached_score": args.data_se3_so3_use_cached_score,
                },
            },
            "dp_cfg": (
                {"dp_root": args.data_dp_data_root}
                if args.data_dp_data_root is not None and args.data_dp_data_root.lower() != "none"
                else None
            )
        },
        "model_cfg": {
            "model_name": args.model_name,
            "ckpt_path": args.model_ckpt_path,
            "model_nn": {
                "embedder": {
                    "time_emb_size": args.model_nn_embedder_time_emb_size,
                    "scale_t": args.model_nn_embedder_scale_t,
                    "res_idx_emb_size": args.model_nn_embedder_res_idx_emb_size,
                    "r_max": args.model_nn_embedder_r_max,
                    "num_rbf": args.model_nn_embedder_num_rbf,
                    "rbf_min": args.model_nn_embedder_rbf_min,
                    "rbf_max": args.model_nn_embedder_rbf_max,
                    "pretrained_node_repr_size": 384,
                    "pretrained_edge_repr_size": 128,
                    "node_emb_size": args.model_nn_embedder_node_emb_size,
                    "edge_emb_size": args.model_nn_embedder_edge_emb_size,
                    "dp_repr_size": args.model_nn_embedder_dp_repr_size, 
                    "use_dp_repr": args.use_dp_repr,
                    "use_af3_relative_pos_encoding": args.model_nn_embedder_use_af3_relative_pos_encoding,
                },
                "structure_module": {
                    "num_ipa_blocks": args.model_nn_structure_module_num_ipa_blocks,
                    "c_s": args.model_nn_structure_module_c_s,
                    "c_z": args.model_nn_structure_module_c_z,
                    "c_hidden": args.model_nn_structure_module_c_hidden,
                    "c_skip": args.model_nn_structure_module_c_skip,
                    "no_heads": args.model_nn_structure_module_no_heads,
                    "no_qk_points": args.model_nn_structure_module_no_qk_points,
                    "no_v_points": args.model_nn_structure_module_no_v_points,
                    "seq_tfmr_num_heads": args.model_nn_structure_module_seq_tfmr_num_heads,
                    "seq_tfmr_num_layers": args.model_nn_structure_module_seq_tfmr_num_layers,
                },
            },
            "se3_cfg": {
                "diffuse_trans": args.model_se3_diffuse_trans,
                "diffuse_rot": args.model_se3_diffuse_rot,
                "r3": {"min_b": args.model_se3_r3_min_b, "max_b": args.model_se3_r3_max_b,
                       "coordinate_scaling": args.model_se3_r3_coordinate_scaling},
                "so3": {
                    "num_omega": args.model_se3_so3_num_omega,
                    "num_sigma": args.model_se3_so3_num_sigma,
                    "min_sigma": args.model_se3_so3_min_sigma,
                    "max_sigma": args.model_se3_so3_max_sigma,
                    "schedule": args.model_se3_so3_schedule,
                    "cache_dir": args.model_se3_so3_cache_dir,
                    "use_cached_score": args.model_se3_so3_use_cached_score,
                },
            },
            "loss": {
                "rot_loss_weight": args.model_loss_rot_loss_weight,
                "rot_angle_loss_t_filter": args.model_loss_rot_angle_loss_t_filter,
                "trans_loss_weight": args.model_loss_trans_loss_weight,
                "bb_coords_loss_weight": args.model_loss_bb_coords_loss_weight,
                "bb_coords_loss_t_filter": args.model_loss_bb_coords_loss_t_filter,
                "bb_dist_map_loss_weight": args.model_loss_bb_dist_map_loss_weight,
                "bb_dist_map_loss_t_filter": args.model_loss_bb_dist_map_loss_t_filter,
                "torsion_loss_weight": args.model_loss_torsion_loss_weight,
                "fape_loss_weight": args.model_loss_fape_loss_weight,
                "clash_loss_weight": args.model_loss_clash_loss_weight,
                "bb_dist_map_cutoff": args.model_loss_bb_dist_map_cutoff,
                "violation_config": {
                    "violation_tolerance_factor": args.model_loss_violation_config_violation_tolerance_factor,
                    "clash_overlap_tolerance": args.model_loss_violation_config_clash_overlap_tolerance,
                    "eps": args.model_loss_violation_config_eps,
                    "weight": args.model_loss_violation_config_weight,
                }
            },
            "reverse_sample_cfg": {
                "num_samples": args.model_reverse_sample_num_samples,
                "scale_coords": args.model_reverse_sample_scale_coords,
                "diffusion_steps": args.model_reverse_sample_diffusion_steps,
                "is_show_diffusing": args.model_reverse_sample_is_show_diffusing,
            },
        },
        "train_cfg": {
            "Pretrain_ckpt_path": args.train_Pretrain_ckpt_path,
            "optimizer": {
                "_target_": torch.optim.AdamW,
                "_partial_": True,
                "lr": args.train_optimizer_lr,
                "weight_decay": args.train_optimizer_weight_decay,
            },
            "scheduler": {
                "_target_": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "_partial_": True,
                "mode": args.train_scheduler_mode,
                "factor": args.train_scheduler_factor,
                "patience": args.train_scheduler_patience,
                "threshold": args.train_scheduler_threshold,
                "min_lr": args.train_scheduler_min_lr,
            },
            "max_epochs": args.train_max_epochs,
            "devices": args.train_devices,
            "strategy": DDPStrategy(gradient_as_bucket_view=True),
            "precision": args.train_precision,
            "log_every_n_steps": args.train_log_every_n_steps,
            "lr_warmup_steps": args.train_lr_warmup_steps,
            "accelerator": args.train_accelerator,
            "val_gen_every_n_epochs": args.train_val_gen_every_n_epochs,
            "accumulate_grad_batches": args.train_accumulate_grad_batches,
            "gradient_clip_val": args.train_gradient_clip_val,
            "gradient_clip_algorithm": args.train_gradient_clip_algorithm,
            "deterministic": args.train_deterministic,
            "inference_mode": args.train_inference_mode,
            "log_path": args.train_log_path,
            "save_ckpt_path": args.save_ckpt_path,
        }
    }

    return config["data_cfg"], config["model_cfg"], config["train_cfg"]
