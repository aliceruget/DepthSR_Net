# DepthSR_Net 

Run :

- *data_path* can be data for the training or the testing.
- *is_train* is a boolean. True for training. False for testing.
- *checkpoint_dir*

`python main.py --data_path = ''  --is_train = '' --config= ''--checkpoint_dir= '' --result_path='' --save_parameters = ''`

# Sources
https://li-chongyi.github.io/proj_SR.html


python main.py --data_path='/Users/aliceruget/Documents/PhD/Dataset/Middlebury_dataset/2005/Art/DATA_TEST/16' --is_train='0' --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Configs/cfg_original_scale16.yaml' --checkpoint_dir='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Checkpoint/16' --result_path='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Results/DATA_TEST_scale16'



python main.py --data_path='/Users/aliceruget/Documents/PhD/Dataset/MPI_Sintel_Depth/depth_16_test/DATA_TRAIN' --is_train='1' --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Configs/cfg_original_scale16.yaml' --checkpoint_dir='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Checkpoint/Training/16'  --result_path='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Training' --save_parameters='0'

cfg_original_scale16.yaml

python3 main.py --data_path='/home/alice_ruget/DATA_TRAIN' --is_train='1' --config='/home/alice_ruget/Configs/cfg_original_scale16.yaml' --checkpoint_dir='/home/alice_ruget/Checkpoint/Training/16' --result_path='/home/alice_ruget/Training'  --save_parameters='0'


link for parallelize 
https://medium.com/@ntenenz/distributed-tensorflow-2bf94f0205c3
dep