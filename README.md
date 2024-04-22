**Gal-DINO**
========

[Journal Paper](https://doi.org/10.1017/pasa.2023.64)

[NeurIPS 2023](https://nips.cc/virtual/2023/76102)

## Installation
Create a Python 3.10.9 environement with CUDA 11.6.2.
Then, install PyTorch 1.5.1+ and torchvision 0.6.1+:
```
conda create -n gal-dino python==3.10.9
conda activate gal-dino
```
For Pawsey Setonix continue with (as of 22.4.2024):
```
module load rocm/5.6.1 gcc/12.2.0  gromacs-amd-gfx90a/2023.2
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

Otherwise:
```
conda install -c pytorch pytorch torchvision
```
  
In both cases install packages in requirements.txt.
```
conda install conda-forge::opencv
pip install -r requirements.txt
```

### Compiling CUDA operators
```
cd models/dino/ops
CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include/ python setup.py build
python setup.py install
# unit test (should see all checking is True). Note you may run out of GPU memory so adjust 'for channels in [..]' in test.py accordingly
python test.py
cd -
```

## Data preparation

Download and extract RadioGalaxyNET data from [here](https://doi.org/10.25919/btk3-vx79) (for Pawsey Setonix follow the link and select s3 rclone method after clicking download. You will get a command to paste into Setonix term).
We expect the directory structure to be the following:
```
./RadioGalaxyNET/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
  test/     # test images
```

## Training

To train on a single node with single gpu run:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py -c config/DINO/DINO_4scale.py
```
To ease reproduction of our results we provide model checkpoint [here](https://figshare.com/s/01dd33b8ff14ffc32dd5). 
Place the model in `./outputs_gal/` directory.

## Evaluation
To evaluate on test images with a single GPU run:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py -c config/DINO/DINO_4scale.py --eval --resume outputs_gal/checkpoint.pth
```

## License
Apache 2.0 license.

