accelerate launch --num_cpu_threads_per_process 16 train.py num_workers=16 eval_overlap=0.7 \
data=task3 data.root=/workspace/dataset/Task03_Liver \
name=medical_decathlon_task3 \
model=swin_unetr optimizer=sgd scheduler=warmup_cosine \
optimizer.params.lr=5.0e-4 optimizer.params.weight_decay=1.0e-4 \
batch_size.train=4 batch_size.val=4 accumulation_steps=1 \
val_freq=10 save_tag=swin_unetr_task03
