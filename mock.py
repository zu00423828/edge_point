from glob import glob
from tkinter.font import names
import pandas as pd

# ffmpeg -i video frames/%06d.jpg
#刪除非場地的畫面

def create_mock():
    with open('data.csv', 'w+') as f:
        for item in glob('frames/*'):
            f.write(f'{item},420,244,298,592,983,589,860,240\n')


create_mock()
df = pd.read_csv('data.csv', names=[
                 'path', 'p1_x', 'p1_y', 'p2_x', 'p2_y''p3_x''p3_y''p4_x''p4_y'])
