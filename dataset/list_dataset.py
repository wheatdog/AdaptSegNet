import glob
import os
import numpy as np

from torch.utils import data
from PIL import Image

class ListDataSet(data.Dataset):
    def __init__(self, root, img_list, lbl_list, max_iters=None, split='train',
                 crop_size=(321, 321), mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.split = split

        self.crop_size = crop_size

        self.ignore_label = ignore_label
        self.mean = mean

        self.files = []

        with open(img_list) as img_files, open(lbl_list) as lbl_files:
            for img_filepath, lbl_filepath in zip(img_files, lbl_files):
                img_filepath = img_filepath.strip()
                lbl_filepath = lbl_filepath.strip()

                name = os.path.basename(img_filepath)

                img_file = os.path.join(self.root, img_filepath)
                lbl_file = os.path.join(self.root, lbl_filepath)

                self.files.append({
                    'image': img_file,
                    'label': lbl_file,
                    'name': name,
                })

        if not max_iters==None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]


        image = Image.open(datafiles["image"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        size = (image.size[1], image.size[0])

        # resize

        image = image.resize(self.crop_size, Image.BICUBIC)

        # This is Hack
        if self.split != 'val':
            label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
