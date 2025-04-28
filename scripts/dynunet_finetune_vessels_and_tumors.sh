accelerate launch --num_cpu_threads_per_process 16 train.py \
num_epochs=10000 num_workers=16 eval_overlap=0.7 name=vessels_and_tumors \
data=ntuh buffer_size=8 \
optimizer=sgd scheduler=cyclic \
optimizer.params.lr=1.0e-4 optimizer.params.weight_decay=1.0e-3 \
scheduler.params.base_lr=1e-8 scheduler.params.max_lr=1e-4 \
batch_size.train=8 batch_size.val=24 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task08_finetune load_from_artifact=true \
loss_fn.params.lambda_ce=1.5 \
load_tag='luisleo52655/vessels_and_tumors/model-best:DynUNet'