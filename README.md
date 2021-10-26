# Network for Classification
This repository contains files for 2D densenet based classification CNN model file and training file.
## Libraries: matplotlib, pytorch, pandas, h5py

## data: 
- training: h5file containing dataset 'data', 'label'. 'data' in shape N x D x H x W
- testing: h5file 

## Training:
- run `python main_2d.py`
- Arguments:
  - `--path_data_tr`: the path to the h5file storing the training data
  - `--path_data_te`: the path to the h5file storing the testing data
  - `--batchsize`: the batch size 
  - `--lr`: initial learning rate
  - `--step`: the number of epochs for keeping the same learning rate
  - `--lr_reduce`: the rate of reduction of learning rate
  - `--num_out`: number of outputs of the classification network
  - `--block_config`: in the form of (4,4,4,4). The number of layers for each block.
  - `--cuda`: select which cuda to use, usually '0', '1' or '0,1'
  - `--ID`: the ID of the folder where the trained models are saved
  - `--resume`: the path of the checkpoint which you want to resume training


## Results:
Results of validation/testing of the model trained after each epoch is saved in folder of checkpoints.
