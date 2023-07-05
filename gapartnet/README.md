- Run the code following the instruction in the README.md in the upper folder.

- We publish our checkpoint, follow the dataset download instructions in the upper folder.


## How to use our code and model: 

### 1. Install dependencies
  - Python 3.8
  - Pytorch >= 1.11.0
  - CUDA >= 11.3
  - Open3D with extension (See install guide below)
  - epic_ops (See install guide below)
  - pointnet2_ops (See install guide below)
  - other pip packages

### 2. Install Open3D & epic_ops & pointnet2_ops
  See this repo for more details:
  
  [GAPartNet_env](https://github.com/geng-haoran/GAPartNet_env): This repo includes Open3D, [epic_ops](https://github.com/geng-haoran/epic_ops) and pointnet2_ops. You can install them by following the instructions in this repo.

### 3. Download our model and data
  See gapartnet folder for more details.

### 4. Inference and visualization
  ```
  cd gapartnet

  CUDA_VISIBLE_DEVICES=0 \
  python train.py test -c gapartnet.yaml \
  --model.init_args.ckpt ckpt/release.ckpt
  ```
Notice:
- We provide visualization code here, you can change cfg in `model.init_args.visualize_cfg` and can control what to visualize (save_option includes ["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt", "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure", "bbox_pred", "bbox_pred_pure"]) and the number of visualization samples.
- We fix some bugs for mAP computation, check the code for more details.

### 5. Training
  You can run the following code to train the policy:
  ```
  cd gapartnet

  CUDA_VISIBLE_DEVICES=0 \
  python train.py fit -c gapartnet.yaml
  ```
Notice:
- For training, please use a good schedule, first train the semantic segmentation backbone and head, then, add the clustering and scorenet supervision for instance segmentation. You can change the schedule in cfg(`model.init_args.training_schedule`). The schedule is a list, the first number indicate the epoch to start the clustering and scorenet training, the second number indicate the epoch to start the npcsnet training. For example, [5,10] means that the clustering and scorenet training will start at epoch 5, and the npcsnet training will start at epoch 10.
- If you want to debug, add `--model.init_args.debug True` to the command and also change `data.init_args.xxx_few_shot` in the cfg to be `True`, here `xxx` is the name of training and validation sets.
- We also provide multi-GPU parallel training, please set `CUDA_VISIBLE_DEVICES` to be the GPUs you want to use, e.g. `CUDA_VISIBLE_DEVICES=3,6` means you want to use 2 GPU #3 and #6 for training.
  