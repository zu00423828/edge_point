from glob import glob
import pandas as pd

# ffmpeg -i video frames/%06d.jpg
# 刪除非場地的畫面


def create_mock():
    with open('data.csv', 'w+') as f:
        for item in glob('frames/*'):
            f.write(f'{item},420,244,298,592,983,589,860,240\n')


if __name__ == '__main__':
    create_mock()
