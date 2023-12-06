accelerate launch --num_cpu_threads_per_process 16 train.py \
data=task3 data.root=/workspace/dataset/Task03_Liver \
name=medical_decathlon_task3 \
model=dynunet optimizer=adam scheduler=warmup_cosine \
optimizer.params.lr=1.0e-2 \
batch_size.train=16 batch_size.val=24 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task03