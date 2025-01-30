accelerate launch --num_cpu_threads_per_process 16 train.py \
num_workers=16 eval_overlap=0.7 name=Tumors \
data=tumors val_portion=0.15 \
optimizer=adam optimizer.params.lr=1.0e-4 optimizer.params.weight_decay=1.0e-2 \
batch_size.train=4 batch_size.val=24 accumulation_steps=1 \
loss_fn.params.include_background=true \
val_freq=10 save_tag=dynunet_1st_tumors
