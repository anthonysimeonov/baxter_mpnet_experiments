python torch_train.py \
--env_data_path ./env/environment_data/ --path_data_path ./data/train/paths/ --pointcloud_data_path ./data/train/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainPaths.pkl \
--trained_model_path ./models/sample/voxel5_64_001/ \
--batch_size 100 --learning_rate 0.001  --num_epochs 400 \
--enc_input_size 64 --enc_vox_size 64 --enc_output_size 128 --mlp_input_size 142 --mlp_output_size 7 --AE_type voxel5 --N 1 --NP 100 \
--device 2 --exp_name voxel5
