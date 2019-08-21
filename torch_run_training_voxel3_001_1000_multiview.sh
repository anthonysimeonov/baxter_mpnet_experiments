python torch_train.py \
--env_data_path ./env/environment_data/ --path_data_path ./data/train/paths/ --pointcloud_data_path ./multiview_data/train/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainPaths.pkl \
--trained_model_path ./models/sample/voxel3_001_1000_multiview/ \
--batch_size 100 --learning_rate 0.001  --num_epochs 200 \
--enc_input_size 32 --enc_output_size 128 --mlp_input_size 142 --mlp_output_size 7 --AE_type voxel3 --N 10 --NP 1000 \
--device 1 --exp_name voxel3
