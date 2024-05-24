accelerate launch --num_cpu_threads_per_process 8 train.py \
dataset.params.num_workers=8 num_workers=8 buffer_size=8 eval_overlap=0.7 \
data=task4 data.root=/workspace/dataset/Task04_Hippocampus \
name=MSD_task04 slices_to_show=5 loss_fn.params.batch=false \
model.params.deep_supr_num=1 \
optimizer=adam optimizer.params.lr=1.0e-3 optimizer.params.weight_decay=1.0e-2 \
batch_size.train=2 batch_size.val=4 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task04