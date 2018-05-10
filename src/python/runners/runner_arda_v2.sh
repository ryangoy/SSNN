python main.py --num_kernels=16 --rotated_copies=0 --num_epochs=400 --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford --loc_loss_lambda=4
python compute_bbox_accuracy.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/ll4.txt

