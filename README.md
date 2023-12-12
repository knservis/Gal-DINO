**Gal-DINO**
========

## Installation
Create a Python 3.10.9 environement with CUDA 11.6.2.
Then, install PyTorch 1.5.1+ and torchvision 0.6.1+:
```
conda install -c pytorch pytorch torchvision
```
  
Install packages in requirements.txt.
```
pip install -r requirements.txt
```

### Compiling CUDA operators
```
cd models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

## Data preparation

Download and extract RadioGalaxyNET data from [here](https://doi.org/10.25919/btk3-vx79).
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

