# DepthSR_Net 

Run :

- *data_path* can be data for the training or the testing.
- *is_train* is a boolean. True for training. False for testing.
- *checkpoint_dir*

`python main.py --data_path = ''  --is_train = '' --config= ''--checkpoint_dir= '' --result_path='' --save_parameters = ''`

# Sources
https://li-chongyi.github.io/proj_SR.html


python main.py --data_path='/Users/aliceruget/Documents/PhD/Codes_HierarchiaclFeatures/Papers_version/depth_16_test/DATA_TEST_original' --is_train='0' --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Configs/cfg_original_scale16.yaml' --checkpoint_dir='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Checkpoint/16' --result_path='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Results/original_DATA_TEST_scale16'

python main.py --data_path='/Users/aliceruget/Documents/PhD/Dataset/MPI_Sintel_Depth/depth_16_test/DATA_TRAIN' --is_train='1' --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Configs/cfg_original_scale16.yaml' --checkpoint_dir='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Checkpoint/Training/16'  --result_path='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Training' --save_parameters='0'