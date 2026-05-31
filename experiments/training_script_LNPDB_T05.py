import subprocess
import os
import shutil

# Finetuning on the public LNPDB dataset (experiments/data_json/LNPDB.json).
# Same recipe as training_script_LANCE_lipid_T01.py, pointed at the LNPDB folds
# and the LNPDB task schema (12 labels, multi-task).
#
# Base weights: the pretrained UniMol molecular encoder (mol_pre_no_h_220816.pt).
# The 8-layer LNP-composition transformer is trained fresh on LNPDB.

data_path='./'  # replace with your data path
MASTER_PORT=10086
n_gpu=1
dict_name='dict.txt'
weight_path='../ckp/mol_pre_no_h_220816.pt'  # pretrained molecular encoder
task_num=1
local_batch_size=32
only_polar=0 # -1 all h; 0 no h
conf_size=11

# LNPDB k-fold data produced by preprocess_data_LNPDB.py
root_task_name='processed_data_dirs/lnpdb_finetune_gen/fold_V0'  # data folder name
root_save_dir='./save_lnpdb'  # replace with your save path
root_tmp_save_dir = './tmp_save_lnpdb'

# Variables To Change during Grid Search:
metric="valid_spearmanr_coeff" # or "valid_spearmanr_coeff"
lr=1e-5
batch_size=32
local_batch_size=batch_size
update_freq=batch_size / local_batch_size
warmup=0.06
dropout=0.1
loss_sample_dropout=0.2
epoch=200

lnp_encoder_attention_heads_list = [8]
lnp_encoder_ffn_embed_dim_list = [256]
lnp_encoder_embed_dim_list = [256]
lnp_encoder_layers_list = [8]
warmups=[0.06]
dropouts=[0.1]
epoch_list = [200]
lrs = [1e-4]
batch_sizes = [64]
loss_sample_dropouts = [0]
loss_funcs = ['np_finetune_contrastive']
full_dataset_task_schema_path = "task_schemas/lnpdb_master_schema.json"
patiences = [20] # train until end of epoch
subdataset_patiences = [-1]
epoch_to_freeze_molecule_encoder_list = [1000000]
cagrad_cs = [0.2]
percent_noises = [0.1]
contrast_margin_coeffs = [0.01]
percent_noise_types = ['normal_proportionate']
save_all_model_weights = True
train_data_ratios = [1]
# Start with a single seed for a first finetune run; expand to a sweep
# (e.g. [1,2,3,...,20]) once the pipeline is validated.
seeds=[1]

for seed in seeds:
    for lnp_encoder_attention_heads in lnp_encoder_attention_heads_list:
        for lnp_encoder_ffn_embed_dim in lnp_encoder_ffn_embed_dim_list:
            for lnp_encoder_embed_dim in lnp_encoder_embed_dim_list:
                for lnp_encoder_layers in lnp_encoder_layers_list:
                    for warmup in warmups:
                        for dropout in dropouts:
                            for epoch in epoch_list:
                                for lr in lrs:
                                    for batch_size in batch_sizes:
                                        for loss_sample_dropout in loss_sample_dropouts:
                                            for loss_func in loss_funcs:
                                                for cagrad_c in cagrad_cs:
                                                    for epoch_to_freeze_molecule_encoder in epoch_to_freeze_molecule_encoder_list:
                                                        for subdataset_patience in subdataset_patiences:
                                                                for contrast_margin_coeff in contrast_margin_coeffs:
                                                                    for percent_noise in percent_noises:
                                                                        for percent_noise_type in percent_noise_types:
                                                                            for patience in patiences:
                                                                                for train_data_ratio in train_data_ratios:
                                                                                    task_name = root_task_name

                                                                                    # set up batch_size
                                                                                    local_batch_size=batch_size
                                                                                    update_freq=batch_size / local_batch_size

                                                                                    # compensate max_epoch with loss_sample_dropout
                                                                                    max_epoch = int(epoch // (1 - loss_sample_dropout))

                                                                                    # unique experiment name (identifier)
                                                                                    exp_name=f'lnpdb_finetune_fold_V0_lnp_{loss_func}-bs{batch_size}-lr{lr}-lnpmodparams{lnp_encoder_layers}-{lnp_encoder_embed_dim}-{lnp_encoder_ffn_embed_dim}-{lnp_encoder_attention_heads}-trainrat{train_data_ratio}-ep{max_epoch}-pat{patience}-metric{metric}-cagrad{cagrad_c}-percentnoise{percent_noise}-labelmargin{contrast_margin_coeff}-seed{seed}'

                                                                                    print("task_name: ", task_name)
                                                                                    if save_all_model_weights:
                                                                                        save_path = 'save_' + exp_name
                                                                                        save_dir = os.path.join(root_save_dir, save_path)
                                                                                        tmp_save_dir = os.path.join(root_tmp_save_dir, save_path)
                                                                                    else:
                                                                                        save_dir = os.path.join(root_save_dir, task_name)
                                                                                        tmp_save_dir = os.path.join(root_tmp_save_dir, task_name)

                                                                                    # tensorboard log path
                                                                                    log_path = 'log_' + exp_name
                                                                                    logdir=os.path.join("./logs/tmp/", log_path)


                                                                                    # infer output path
                                                                                    results_folder = 'infer_' + exp_name
                                                                                    eval_results_path = os.path.join("./infer_results/", results_folder)
                                                                                    eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')


                                                                                    # Check if this experiment is already done, if so, skip it
                                                                                    if os.path.exists(eval_results_path):
                                                                                        print("Infer output dir exists. Skipping experiment: ", eval_results_path)
                                                                                        continue
                                                                                    elif os.path.exists(logdir): # if logdir exists, delete it as the previous exp run is not done yet
                                                                                        print("Tensorboard log dir exists but inference not done. Rerunning exp and removing log dir: ", logdir)
                                                                                        shutil.rmtree(logdir)

                                                                                    if os.path.exists(save_dir) and os.path.isdir(save_dir):
                                                                                        shutil.rmtree(save_dir)

                                                                                    if os.path.exists(tmp_save_dir) and os.path.isdir(tmp_save_dir):
                                                                                        shutil.rmtree(tmp_save_dir)


                                                                                    print("Running training script for: ", logdir)

                                                                                    subprocess.run(f"python ../unimol/train_np.py {data_path} --task-name {task_name} --user-dir ../unimol --train-subset train --valid-subset valid \
                                                                                        --conf-size {conf_size} \
                                                                                        --num-workers 8 --ddp-backend=c10d \
                                                                                        --dict-name {dict_name} \
                                                                                        --task mol_np_finetune --loss {loss_func} --arch np_unimol  \
                                                                                        --classification-head-name {task_name} --num-classes {task_num} \
                                                                                        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                                                                                        --lr-scheduler polynomial_decay --lr {lr} --warmup-ratio {warmup} --max-epoch {max_epoch} --batch-size {local_batch_size} --pooler-dropout {dropout} \
                                                                                        --loss-sample-dropout {loss_sample_dropout} \
                                                                                        --update-freq {update_freq} --seed {seed} \
                                                                                        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                                                                                        --log-interval 100 --log-format simple \
                                                                                        --validate-interval 1 --keep-last-epochs 10 \
                                                                                        --finetune-from-model {weight_path} \
                                                                                        --best-checkpoint-metric {metric} --patience {patience} \
                                                                                        --maximize-best-checkpoint-metric \
                                                                                        --save-dir {save_dir} --tmp-save-dir {tmp_save_dir} --only-polar {only_polar} \
                                                                                        --tensorboard-logdir {logdir} \
                                                                                        --full-dataset-task-schema-path {full_dataset_task_schema_path} \
                                                                                        --multitask-reg --cagrad-c {cagrad_c} \
                                                                                        --epoch-to-freeze-molecule-encoder {epoch_to_freeze_molecule_encoder} \
                                                                                        --concat-datasets \
                                                                                        --train-data-ratio {train_data_ratio} \
                                                                                        --lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} \
                                                                                        --noise-augment-percent --percent-noise {percent_noise} --percent-noise-type {percent_noise_type} \
                                                                                        --contrast-margin-coeff {contrast_margin_coeff}",
                                                                                        shell=True)

                                                                                    # eval params
                                                                                    eval_batch_size = 32

                                                                                    # tensorboard log path
                                                                                    results_folder = 'infer_' + exp_name
                                                                                    eval_results_path = os.path.join("./infer_results/", results_folder)
                                                                                    eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')

                                                                                    subprocess.run(f"python ../unimol/infer_np.py --user-dir ../unimol {data_path} --task-name {task_name} --valid-subset test \
                                                                                        --num-workers 8 --ddp-backend=c10d --batch-size {eval_batch_size} \
                                                                                        --task mol_np_finetune --loss {loss_func} --arch np_unimol \
                                                                                        --classification-head-name {task_name} --num-classes {task_num} \
                                                                                        --dict-name {dict_name} --conf-size {conf_size} \
                                                                                        --only-polar {only_polar}  \
                                                                                        --path {eval_weight_path}  \
                                                                                        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                                                                                        --log-interval 50 --log-format simple \
                                                                                        --results-path {eval_results_path} \
                                                                                        --lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} \
                                                                                        --full-dataset-task-schema-path {full_dataset_task_schema_path} \
                                                                                        --load-full-np-model --concat-datasets",
                                                                                        shell=True)
