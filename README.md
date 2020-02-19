# DepthSR_Net 

Input

- *data_path* can be data for the training or the testing.
- *is_train* is a boolean. True for training. False for testing.
- *checkpoint_dir*
- *result_path* 
- *save_parameters* is a boolean. Set to 1 to save filters, bias, feature maps. 
- *config* configuration of network. 

Output :
- *configuration.mat* with data_path, result_path, config, is_train
- *parameters.mat* (if save_parameters = 1)
- *RMSE_tab* save the evolution of RMSE at each step
- *_down* : input depth 
- *_rgb* : input intensity 
- *_up* : GT depth 
- *_sr* : output of algorithm

`python main.py --data_path = ''  --is_train = '' --config= ''--checkpoint_dir= '' --result_path='' --save_parameters = ''`

# Sources

https://li-chongyi.github.io/proj_SR.html


# Example

python main.py --data_path='/Users/aliceruget/Documents/PhD/Dataset/Paper_Test_dataset/16' --is_train='0' --config='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Configs/cfg_original_scale16.yaml' --checkpoint_dir='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Checkpoint/16' --result_path='/Users/aliceruget/Documents/PhD/DepthSR_Net_AR/Results/Paper_Test_dataset_16' --save_parameters='1'
