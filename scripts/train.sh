accelerate launch --num_cpu_threads_per_process 8 train.py \
data.root=/workspace/dataset/Task08_HepaticVessel \
model.batch_size.train=2 model.batch_size.val=16 \
model.optimizer.params.lr=1e-2 \
model.scheduler.params.base_lr=1e-3 model.scheduler.params.max_lr=1e-2 \
val_freq=10 \
save_tag=train_from_scratch