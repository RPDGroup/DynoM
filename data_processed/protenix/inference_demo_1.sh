
export PYTHONPATH=".:$PYTHONPATH"
export LAYERNORM_TYPE=False
export USE_DEEPSPEED_EVO_ATTTENTION=False

N_sample=5
N_step=200
N_cycle=10
seed=101
use_deepspeed_evo_attention=true


input_json_path="/opt/data/private/test_github_dynom/Jqt_Test/DynoM/example/Test_Task/ADK/01_Data_Process/Step2_Representation_Generation/02_JSON_Construction/Output/9R71.json"
#表征输出路径
dump_dir="/opt/data/private/test_github_dynom/Jqt_Test/DynoM/example/Test_Task/ADK/01_Data_Process/Step2_Representation_Generation/04_Representaion_Generation"


only_encoder=True

torchrun --nproc_per_node=1 runner/inference.py \
--seeds ${seed} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step} \
--only_encoder ${only_encoder}
