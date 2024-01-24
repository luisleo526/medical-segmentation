accelerate launch --num_cpu_threads_per_process 8 train.py num_workers=8 eval_overlap=0.7 \
data=task3 data.root=/workspace/dataset/Task03_Liver \
name=medical_decathlon_task3 \
optimizer=sgd optimizer.params.lr=1.0e-4 optimizer.params.weight_decay=1.0e-3 \
batch_size.train=8 batch_size.val=8 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task03_finetune load=true load_tag=dynunet_task03
