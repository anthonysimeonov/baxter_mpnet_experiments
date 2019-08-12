python torch_train.py \
--env_data_path ./env/environment_data/ --path_data_path ./data/train/paths/ --pointcloud_data_path ./data/train/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainPaths.pkl \
--trained_model_path ./models/sample/voxel2_128_001/ \
--batch_size 100 --learning_rate 0.001  --num_epochs 200 \
--enc_input_size 128 --enc_output_size 128 --mlp_input_size 142 --mlp_output_size 7 --AE_type voxel2 --N 1 --NP 1000 \
--device 0 --exp_name voxel2
