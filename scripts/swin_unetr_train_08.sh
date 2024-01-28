accelerate launch --num_cpu_threads_per_process 16 train.py num_workers=16 eval_overlap=0.7 \
data.root=/workspace/dataset/Task08_HepaticVessel \
model=swin_unetr optimizer=sgd scheduler=warmup_cosine \
optimizer.params.lr=1.0e-2 optimizer.params.weight_decay=3.0e-3 \
batch_size.train=16 batch_size.val=16 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task08