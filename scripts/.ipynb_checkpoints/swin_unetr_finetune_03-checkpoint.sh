accelerate launch --num_cpu_threads_per_process 16 train.py num_workers=16 eval_overlap=0.7 \
num_epochs=3000 \
data=task3 data.root=/workspace/dataset/Task03_Liver \
name=MSD_task03 \
model=swin_unetr optimizer=sgd scheduler=cyclic \
optimizer.params.lr=1.0e-2 optimizer.params.weight_decay=3.0e-3 \
scheduler.params.base_lr=1e-6 scheduler.params.max_lr=1e-2 \
batch_size.train=4 batch_size.val=4 accumulation_steps=4 \
val_freq=10 save_tag=dynunet_task03_finetune load_from_artifact=true \
load_tag='luisleo52655/MSD_task03/model-best:SwinUNETR'
