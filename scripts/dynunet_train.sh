accelerate launch --num_cpu_threads_per_process 8 train.py \
data.root=/workspace/dataset/Task08_HepaticVessel model=dynunet \
model.batch_size.train=4 model.batch_size.val=24 model.accumulation_steps=1 \
model.optimizer.params.lr=1e-2 model.scheduler.params.base_lr=3e-4 model.scheduler.params.max_lr=1e-2 \
val_freq=10 save_tag=dynunet_task08