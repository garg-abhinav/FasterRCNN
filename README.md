# Faster R-CNN Implementation
## IE534/CS547 Deep Learning | Spring 2021 | UIUC

## Contributors
1. Abhinav Garg (garg19@illinois.edu)
2. Refik Mert Cam (rcam2@illinois.edu)
3. Sanyukta Deshpande (spd4@illinois.edu)

## Install dependencies
Here is an example of create environ **from scratch** with `anaconda`

```sh
# create conda env
conda create --name simp python=3.7
conda activate simp
# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet
