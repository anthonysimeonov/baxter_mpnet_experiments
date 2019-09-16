python torch_test.py \
--env_data_path ./env/environment_data/ --path_data_path ./test_data/test/paths/ --pointcloud_data_path ./multiview_data/test/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainEnvironments_testPaths.pkl \
--model_path ./models/sample/mvvoxel5_01/ --mlp_model_name mlp_PReLU_ae_dd180.pkl --enc_model_name cae_encoder_180.pkl --experiment_name test_experiment \
--AE_type mvvoxel5 --enc_input_size 32 --enc_output_size 128 --mlp_input_size 142 --mlp_output_size 7 --device 2 --exp_name mvvoxel5 --N 2
