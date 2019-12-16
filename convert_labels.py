from PIL import Image
import numpy as np
import glob
from skimage import io

label_colormap = {
    "Void": (0, 0, 0, 255),
    "Crosswalk": (255, 0, 0, 255),
    "Double White": (0, 255, 0, 255),
    "Double Yellow": (0, 0, 255, 255),
    "Road Curb": (255, 255, 0, 255),
    "Single White": (255, 0, 255, 255),
    "Single Yellow": (0, 255, 255, 255)
}


id2label = [
    'Void',
    'Crosswalk',
    'Double White',
    'Double Yellow',
    'Road Curb',
    'Single White',
    'Single Yellow',
]

label2id = {label: id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]

for img in glob.glob('C:/Users/khali/Downloads/km10k/data/labels/*.png'):
    file = img
    img = Image.open(img)
    label = np.zeros_like(img)
    img = np.array(img)
    label = np.zeros_like(img)
    label = label[:, :, 0]
    for id in range(len(idcolormap)):
        label[np.all(img == np.array(idcolormap[id]), axis=2)] = id
    io.imsave(file, label)
