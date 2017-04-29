import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import numpy as np
import utils
import os

file_path = os.path.dirname(os.path.abspath(__file__))


def make_example(img):
    ex = tf.train.SequenceExample()
    ex.context.feature["img"].bytes_list.value.append(img)
    return ex


def build_examples():
    path_to_save_examples = os.path.join(file_path, utils.back("examples"))
    if not os.path.exists(path_to_save_examples):
        os.makedirs(path_to_save_examples)

    for name in ["train", "val"]:
        # Path where images are
        path = os.path.join(file_path, utils.back("{}2014".format(name)))

        # Number of tfRecords already created
        # The file number to restart from
        writer = tf.python_io.TFRecordWriter(
            os.path.join(path_to_save_examples, "{}".format(name) + ".tfrecords"))
        for index, filename in tqdm(enumerate(os.listdir(path))):
            img = np.array(Image.open(os.path.join(path, filename)))
            img = img.tostring()
            ex = make_example(img)
            writer.write(ex.SerializeToString())

        writer.close()
        break


if __name__ == '__main__':
    build_examples()
