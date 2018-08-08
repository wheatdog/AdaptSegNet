import glob
import os
import numpy as np

from torch.utils import data
from PIL import Image

class ListDataSet(data.Dataset):
    def __init__(self, root, img_list, lbl_list=None, max_iters=None, split='train',
                 crop_size=(321, 321), mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.split = split

        self.crop_size = crop_size

        self.ignore_label = ignore_label
        self.mean = mean

        self.files = []

        img_files = open(img_list)
        for img_filepath in img_files:
            img_filepath = img_filepath.strip()
            name = os.path.basename(img_filepath)
            img_file = os.path.join(self.root, img_filepath)

            self.files.append({
                'image': img_file,
                'name': name,
            })

        if lbl_list:
            lbl_files = open(lbl_list)
            for idx, lbl_filepath in enumerate(lbl_files):
                lbl_filepath = lbl_filepath.strip()
                lbl_file = os.path.join(self.root, lbl_filepath)
                self.files[idx]['label'] = lbl_file

        if not max_iters==None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["image"]).convert('RGB')
        name = datafiles["name"]

        size = (image.size[1], image.size[0])

        # resize

        image = image.resize(self.crop_size, Image.BICUBIC)


        image = np.asarray(image, np.float32)

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        if 'label' in datafiles:
            label = Image.open(datafiles["label"])
            # This is Hack
            if self.split != 'val':
                label = label.resize(self.crop_size, Image.NEAREST)
            label = np.asarray(label, np.float32)
            label = label.copy()
        else:
            # This is Hack
            label = []

        return image.copy(), label, np.array(size), name


if __name__ == '__main__':
    data_dir = '/4TB/ytliou/crossdata_seg/dataset'
    img_list = '/4TB/ytliou/crossdata_seg/list/gta5-all-img.txt'
    lbl_list = '/4TB/ytliou/crossdata_seg/list/gta5-all-lbl.txt'
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    dset = ListDataSet(data_dir, img_list, 
            crop_size=(1024,512), mean=IMG_MEAN)
    for i, batch in enumerate(dset):
        if i == 5:
            break
        img, lbl, osize, name = batch
        print(i, img.shape, lbl, osize, name)
