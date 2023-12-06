accelerate launch --num_cpu_threads_per_process 16 train.py \
data.root=/workspace/dataset/Task08_HepaticVessel \
model=dynunet optimizer=adam scheduler=warmup_cosine \
optimizer.params.lr=3.0e-4 \
batch_size.train=2 batch_size.val=16 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task08