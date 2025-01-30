accelerate launch --num_cpu_threads_per_process 16 train.py \
num_epochs=10000 num_workers=16 eval_overlap=0.7 name=Tumors buffer_size=16 \
data=tumors val_portion=0.15 \
optimizer=sgd scheduler=cyclic \
optimizer.params.lr=1.0e-4    optimizer.params.weight_decay=1.0e-3 \
scheduler.params.base_lr=1e-8 scheduler.params.max_lr=1e-4 \
loss_fn.params.include_background=true \
batch_size.train=4 batch_size.val=24 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_finetune_tumors load_from_artifact=true \
load_tag='luisleo52655/Tumors/model-best:DynUNet'