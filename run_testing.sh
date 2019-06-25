python test.py \
--env_data_path ./env/environment_data/ --path_data_path ./data/test/paths/ --pointcloud_data_path ./data/test/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainEnvironments_testPaths.pkl \
--model_path ./models/sample/ --mlp_model_name mlp_PReLU_ae_dd1.pkl --enc_model_name cae_encoder_1.pkl --experiment_name test_experiment
