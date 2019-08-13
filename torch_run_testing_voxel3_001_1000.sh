python torch_test.py \
--env_data_path ./env/environment_data/ --path_data_path ./test_data/test/paths/ --pointcloud_data_path ./test_data/test/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainEnvironments_testPaths.pkl \
--model_path ./models/sample/voxel3_001_1000/ --mlp_model_name mlp_PReLU_ae_dd120.pkl --enc_model_name cae_encoder_120.pkl --experiment_name test_experiment \
--AE_type voxel3 --enc_input_size 32 --enc_output_size 128 --mlp_input_size 142 --mlp_output_size 7 --device 0 --exp_name voxel3 --N 10
