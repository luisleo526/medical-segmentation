accelerate launch --num_cpu_threads_per_process 16 train.py num_workers=16 eval_overlap=0.7 \
num_epochs=300 \
data.root=/workspace/dataset/Task08_HepaticVessel \
name=MSD_task08 \
model=dynunet optimizer=sgd scheduler=cyclic \
optimizer.params.lr=3.0e-3 optimizer.params.weight_decay=1.0e-3 \
scheduler.params.base_lr=1e-8 scheduler.params.max_lr=3e-3 \
batch_size.train=16 batch_size.val=24 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task08_finetune load_from_artifact=true \
load_tag='luisleo52655/medical_decathlon_task8/model-best:v139'