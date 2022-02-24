from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np


def create_point(x_list, y_list):
    return [[x, y]for x, y in zip(x_list, y_list)]


class EdgeDataset(Dataset):
    def __init__(self, data) -> None:
        df = pd.read_csv(data, names=[
            'path', 'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y'])
        self.img_list = df['path']
        p1_x = df['p1_x']
        p1_y = df['p1_y']
        p2_x = df['p2_x']
        p2_y = df['p2_y']
        p3_x = df['p3_x']
        p3_y = df['p3_y']
        p4_x = df['p4_x']
        p4_y = df['p4_y']
        p1 = create_point(p1_x, p1_y)
        p2 = create_point(p2_x, p2_y)
        p3 = create_point(p3_x, p3_y)
        p4 = create_point(p4_x, p4_y)
        self.label_list = [[p1, p2, p3, p4]
                           for p1, p2, p3, p4 in zip(p1, p2, p3, p4)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        h, w, _ = img.shape
        img = cv2.resize(img, (256, 256))
        img = (img/255).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        label = self.label_list[index]
        label = np.array(label)
        label = (2 * (label / (np.array((h, w))-1))-1)
        return img, label.astype(np.float32)


if __name__ == '__main__':
    d = EdgeDataset('data.csv')
    print(len(d))
