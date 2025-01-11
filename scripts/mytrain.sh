#! /bin/bash

python train_contrastive.py --root_path /home/lihp/prj/DRCT/dataset/DRCT_data/MSCOCO/train2017 \
                            --fake_root_path /home/lihp/prj/DRCT/dataset/DRCT_data/DRCT-2M/real_rec_images/train2017,/home/lihp/prj/DRCT/dataset/DRCT_data/DRCT-2M/fake_images/train2017,/home/lihp/prj/DRCT/dataset/DRCT_data/DRCT-2M/fake_rec_images//train2017 \
                            --dataset_name DRCT-2M \
                            --model_name convnext_base_in22k \
                            --embedding_size 1024 \
                            --input_size 224 \
                            --batch_size 64 \
                            --fake_indexes 2 \
                            --num_epochs 17 \
                            --device_id 5,6 \
                            --lr 0.0001 \
                            --is_amp \
                            --is_crop \
                            --num_workers 12 \
                            --save_flag _drct_amp_crop