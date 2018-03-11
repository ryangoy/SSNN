python main.py --num_kernels=16 --rotated_copies=0 --num_epochs=400 --load_probe_output=True --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/stanford_pointnet_comparison.txt

