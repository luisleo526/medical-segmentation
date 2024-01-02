accelerate launch --num_cpu_threads_per_process 8 train.py num_workers=8 eval_overlap=0.25 \
data=task3 data.root=/workspace/dataset/Task03_Liver \
name=medical_decathlon_task3 \
optimizer.params.lr=1.0e-4 \
batch_size.train=2 batch_size.val=4 accumulation_steps=1 \
val_freq=10 save_tag=dynunet_task03_finetune load=true load_tag=dynunet_task03
