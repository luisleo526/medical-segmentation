accelerate launch --num_cpu_threads_per_process 16 train.py num_workers=16 eval_overlap=0.7 \
num_epochs=3000 \
data.root=/workspace/dataset/Task08_HepaticVessel \
name=MSD_task08 \
model=swin_unetr optimizer=sgd scheduler=cyclic \
optimizer.params.lr=3.0e-4 optimizer.params.weight_decay=3.0e-3 \
scheduler.params.base_lr=1e-6 scheduler.params.max_lr=3e-4 \
batch_size.train=4 batch_size.val=8 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task08_finetune load_from_artifact=true \
load_tag='luisleo52655/model-registry/MSD_task08:v1'
