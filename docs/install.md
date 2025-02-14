# Installation instructions

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n preworld python=3.8 -y
conda activate preworld
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111  -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install mmcv-full.**

```shell
pip install mmcv-full==1.6.0
```

**d. Install mmdet and mmseg.**

```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

**e. Install PreWorld from source code.**

```shell
git clone https://github.com/getterupper/PreWorld.git
cd PreWorld
pip install -r requirements.txt
pip install -v -e .
# python setup.py install
```
