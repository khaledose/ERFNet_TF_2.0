from PIL import Image
import numpy as np
import h5py

def obj2h5(data, h5_file):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=np.array(value))
        f.close()


def h52obj(h5_file):
    data = {}
    with h5py.File(h5_file, 'r') as f:
        for key, value in f.items():
            data[key] = np.array(value)
        f.close()
    return data

def get_colors(labels, n_classes):
    colormap = []
    for label in labels:
        img = Image.open(label)
        colors = img.convert('RGB').getcolors()
        for i in range(len(colors)):
            if colors[i][1] not in colormap:
                colormap.append(colors[i][1])
        if len(colormap) == n_classes:
            break
    colormap.sort()
    return np.array(colormap)
# colormap = [[0, 0, 0], [255, 0, 0], [0, 255, 0],
#             [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

# label_colormap = {
#     "Void": (0, 0, 0, 255),
#     "Crosswalk": (255, 0, 0, 255),
#     "Double White": (0, 255, 0, 255),
#     "Double Yellow": (0, 0, 255, 255),
#     "Road Curb": (255, 255, 0, 255),
#     "Single White": (255, 0, 255, 255),
#     "Single Yellow": (0, 255, 255, 255)
# }
# id2label = [
#     'Void',
#     'Crosswalk',
#     'Double White',
#     'Double Yellow',
#     'Road Curb',
#     'Single White',
#     'Single Yellow',
# ]

# label2id = {label: id for id, label in enumerate(id2label)}
# idcolormap = [label_colormap[label] for label in id2label]
