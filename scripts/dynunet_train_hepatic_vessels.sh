accelerate launch --num_cpu_threads_per_process 16 train.py \
num_workers=16 eval_overlap=0.7 name=NTUH_HepaticVessel \
data=ntuh \
optimizer=adam optimizer.params.lr=3.0e-4 optimizer.params.weight_decay=1.0e-2 \
loss_fn.params.batch=false \
batch_size.train=4 batch_size.val=24 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_1st 
