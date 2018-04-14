import os
import numpy

PATH = os.path.abspath('./DataSet')
IMAGE_PATH = os.path.abspath('./DataSet/image')

if __name__ == '__main__':
    with open(os.path.join(PATH, 'sample.txt'), 'r', encoding='utf8') as file:
        for tmp in file.readlines():
            if len(tmp) == 1:
                print('Dirty data %s' % tmp[0])
                os.remove(os.path.join(IMAGE_PATH, tmp[0]))
            else:
                with open(os.path.join(PATH, 'sample2.txt'), 'a', encoding='utf8') as file2:
                    file2.write(tmp)
