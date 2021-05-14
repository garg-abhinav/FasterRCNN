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
conda create --name frcnn python=3.7
conda activate frcnn
# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet
```

## Train

### 1. Prepare data

#### Pascal VOC2007

1. Download the training, validation, test data and VOCdevkit

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. modify `voc_data_dir` and `voc_test_dir` cfg item in `config/config.py`.

### 2. Set Configuration
Update the parameters in `config/config.py` as per the experiment. 
Update `save_path` to the path where model files are to be stored.
**Note:** Our implementation currently only supports *vgg16* and *resnet101* for `pretrained_model` cfg item.

### 3. Model Training
```bash
python approx_train.py 
```

### Inference
To run inference on select test images, update `train=False` and `save_path` to the path where trained model is located in `config/config.py`.
```bash
python test.py 
```
