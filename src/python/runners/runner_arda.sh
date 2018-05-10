python main.py --load_probe_output=False --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/default.txt
python main.py --num_dot_layers=8 --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/8dot.txt
python main.py --num_dot_layers=32 --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/32dot.txt
python main.py --num_epochs=450 --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/450epochs.txt

python main.py --loc_loss_lambda=0.1 --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy> logs/lllp1.txt
python main.py --jittered_copies=3 --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/jittered3.txt
python main.py --num_kernels=16 --load_probe_output=False --data_dir=/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2 --dataset_name=stanford
python compute_bbox_accuracy_v2.py /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_predictions.npy /home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/bbox_test_labels.npy > logs/nk16.txt

