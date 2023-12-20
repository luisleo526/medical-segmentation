accelerate launch --num_cpu_threads_per_process 16 train.py \
dataset.params.num_workers=16 num_workers=16 buffer_size=16 eval_overlap=0.1 \
data=task3 data.root=/workspace/dataset/Task03_Liver \
name=medical_decathlon_task3 \
model=dynunet optimizer=sgd scheduler=warmup_cosine \
optimizer.params.lr=1.0e-2 \
batch_size.train=4 batch_size.val=16 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task03