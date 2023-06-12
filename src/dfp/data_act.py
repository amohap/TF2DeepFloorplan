from typing import Dict, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf


def convert_one_hot_to_image(
    one_hot: tf.Tensor, dtype: str = "float", act: str = None
) -> tf.Tensor:
    if act == "softmax":
        one_hot = tf.keras.activations.softmax(one_hot)
    [n, h, w, c] = one_hot.shape.as_list()
    im = tf.reshape(tf.keras.backend.argmax(one_hot, axis=-1), [n, h, w, 1])
    if dtype == "int":
        im = tf.cast(im, dtype=tf.uint8)
    else:
        im = tf.cast(im, dtype=tf.float32)
    return im


def _parse_function(example_proto: bytes, size: int = 512) -> Dict[str, tf.Tensor]:
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "boundary": tf.io.FixedLenFeature([], tf.string),
        "room": tf.io.FixedLenFeature([], tf.string),
        "door": tf.io.FixedLenFeature([], tf.string),
        "act_door": tf.io.FixedLenFeature([], tf.string),
        "act_sitt": tf.io.FixedLenFeature([], tf.string),
        "act_lay": tf.io.FixedLenFeature([], tf.string),
        "act_wash": tf.io.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.io.parse_single_example(example_proto, feature)

    # Decode and reshape the images
    for key in parsed_features.keys():
        # Decode the raw bytes
        parsed_features[key] = tf.io.decode_raw(parsed_features[key], out_type=tf.uint8)
        
        # Reshape the tensor into its final shape
        if key == 'image':
            parsed_features[key] = tf.reshape(parsed_features[key], [size, size, 3])  # assuming your images have 3 color channels
        elif key == 'bound' or key == 'room':
            continue
        else:
            parsed_features[key] = tf.reshape(parsed_features[key], [size, size, 1])  # assuming the others are grayscale with 1 channel

    return parsed_features


def decodeAllRaw(
    x: Dict[str, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    image = tf.io.decode_raw(x["image"], tf.uint8)
    boundary = tf.io.decode_raw(x["boundary"], tf.uint8)
    room = tf.io.decode_raw(x["room"], tf.uint8)
    act_door = tf.io.decode_raw(x["act_door"], tf.uint8)
    act_sitt = tf.io.decode_raw(x["act_sitt"], tf.uint8)
    act_lay = tf.io.decode_raw(x["act_lay"], tf.uint8)
    act_wash = tf.io.decode_raw(x["act_wash"], tf.uint8)
    return image, boundary, room, act_door, act_sitt, act_lay, act_wash


def preprocess(
    img: tf.Tensor, bound: tf.Tensor, room: tf.Tensor, act_door: tf.Tensor, act_sitt: tf.Tensor, act_lay: tf.Tensor, act_wash: tf.Tensor, size: int=512
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    
    # Normalize the image to [0, 1] range.
    img = tf.cast(img, dtype=tf.float32) / 255
    
    # No need to reshape since it's already done in _parse_function()
    # img = tf.reshape(img, [-1, size, size, 3]) / 255

    # The other tensors are also already reshaped in _parse_function()
    bound = tf.reshape(bound, [-1, size, size])
    room = tf.reshape(room, [-1, size, size])

    # No need to add extra dimension or resize since it's already done in _parse_function()
    act_door = tf.cast(act_door, tf.float32)
    act_sitt = tf.cast(act_sitt, tf.float32)
    act_lay = tf.cast(act_lay, tf.float32)
    act_wash = tf.cast(act_wash, tf.float32)

    hot_b = tf.one_hot(bound, 3, axis=-1)
    hot_r = tf.one_hot(room, 9, axis=-1)

    # concatenate the activity tensors to the input image along the channel dimension
    img = tf.concat([img, act_door, act_sitt, act_lay, act_wash], axis=-1)

    return img, bound, room, act_door, act_sitt, act_lay, act_wash, hot_b, hot_r



def loadDataset(size: int = 512) -> tf.data.Dataset:
    raw_dataset = tf.data.TFRecordDataset("/cluster/scratch/amohap/data/tf2deep/tf2deep_act_train.tfrecords")
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def plotData(data: Dict[str, tf.Tensor]):
    img, bound, room, act_door, act_sitt, act_lay, act_wash = decodeAllRaw(data)
    img, bound, room, act_door, act_sitt, act_lay, act_wash, hb, hr = preprocess(img, bound, room)
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(bound[0].numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(convert_one_hot_to_image(hb)[0].numpy())


def main(dataset: tf.data.Dataset):
    for ite in range(2):
        for data in list(dataset.shuffle(400).batch(1)):
            plotData(data)
            plt.show()
            break


if __name__ == "__main__":
    dataset = loadDataset()
    main(dataset)
