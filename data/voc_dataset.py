import os
import xml.etree.ElementTree as ET
import numpy as np
from .utils import read_image

VOC_BBOX_LABELS = ('aeroplane',
                   'bicycle',
                   'bird',
                   'boat',
                   'bottle',
                   'bus',
                   'car',
                   'cat',
                   'chair',
                   'cow',
                   'diningtable',
                   'dog',
                   'horse',
                   'motorbike',
                   'person',
                   'pottedplant',
                   'sheep',
                   'sofa',
                   'train',
                   'tvmonitor')


class VOCBboxDataset:
    def __init__(self, data_dir, split='trainval'):
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABELS

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        Returns the i-th example of the dataset.
        Args:
            item (int): The index of the example.
        Returns:
            tuple of an RGB image in CHW format and bounding boxes
        """
        id_ = self.ids[item]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox, label = [], []
        for obj in anno.findall('object'):
            # skip the objects that are difficult
            if int(obj.find('difficult').text) == 1:
                continue

            bbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([int(bbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        img = read_image(os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg'))

        return img, bbox, label
