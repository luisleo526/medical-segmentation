accelerate launch --num_cpu_threads_per_process 16 train.py \
num_workers=16 eval_overlap=0.7 name=vessels_and_tumors \
data=ntuh buffer_size=8 \
loss_fn.params.batch=false loss_fn.params.include_background=false loss_fn.params.lambda_ce=1.5 \
optimizer=adam optimizer.params.lr=3.0e-5 optimizer.params.weight_decay=3.0e-3 \
batch_size.train=8 batch_size.val=24 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_vessels_and_tumors 
