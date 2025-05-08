

## Dependencies

- Python 3.8
- PyTorch 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'SwinGSR'.
git clone https://github.com/chentaoqian/SwinGSR.git
conda create -n SwinGSR python=3.8
conda activate SwinGSR
pip install -r requirements.txt
python setup.py develop
```

## Training
- Run the following scripts. The training configuration is in `options/train/`.
  ```shell
  python basicsr/train.py -opt options/Train/train_SwinGSR_x2.yml
  python basicsr/train.py -opt options/Train/train_SwinGSR_x4.yml
  ```
- The training experiment is in `experiments/`.
  
## Testing
- Run the following scripts. The testing configuration is in `options/test/`.
  ```shell
  python basicsr/train.py -opt options/Test/my_test_SwinGSR_x2.yml
  python basicsr/train.py -opt options/Test/my_test_SwinGSR_x4.yml
  ```
- The output is in `results/`.

## Acknowledgements

This code is built on  [SwinIR]([https://github.com/zhengchen1999/DAT.git](https://github.com/JingyunLiang/SwinIR/tree/main)).

