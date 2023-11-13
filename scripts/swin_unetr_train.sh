accelerate launch --num_cpu_threads_per_process 8 train.py \
data.root=/workspace/dataset/Task08_HepaticVessel model=swin_unetr \
model.batch_size.train=2 model.batch_size.val=16 model.accumulation_steps=1 \
model.optimizer.params.lr=1e-2 \
val_freq=10 save_tag=swin_unetr_task08