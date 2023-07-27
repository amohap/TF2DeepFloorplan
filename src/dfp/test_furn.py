import argparse
import io
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from .data_furn import (
    convert_one_hot_to_image,
    loadDataset,
    preprocess,
)
from .loss import balanced_entropy, cross_three_tasks_weight, cross_two_tasks_weight
from .net import deepfloorplanModel, deepfurnfloorplanModel
from .net_func import deepfloorplanFunc
from .utils.settings import overwrite_args_with_toml


def init(
    config: argparse.Namespace,
) -> Tuple[tf.data.Dataset, tf.keras.Model, tf.keras.optimizers.Optimizer]:
    dataset = loadDataset(config)
    if config.tfmodel == "subclass":
        if config.furniture or config.activities:
            model = deepfurnfloorplanModel(config=config)
        else:
            model = deepfloorplanModel(config=config)
    elif config.tfmodel == "func":
        model = deepfloorplanFunc(config=config)
    os.system(f"mkdir -p {config.modeldir}")
    if config.weight:
        model.load_weights(config.weight)
    # optim = tf.keras.optimizers.AdamW(learning_rate=config.lr,
    #   weight_decay=config.wd)
    optim = tf.keras.optimizers.Adam(learning_rate=config.lr)
    return dataset, model, optim


def plot_to_image(figure: matplotlib.figure.Figure) -> tf.Tensor:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(
    config: argparse.Namespace,
    img: tf.Tensor,
    bound: tf.Tensor,
    room: tf.Tensor,
    furn: tf.Tensor,
    logr: tf.Tensor,
    logcw: tf.Tensor,
    logf: tf.Tensor
) -> matplotlib.figure.Figure:
    figure = plt.figure(figsize=(15,10))
    plt.subplot(2, 5, 1)
    plt.imshow(img[0, :, :, :3].numpy())  # only plot first 3 channels as the original image
    plt.title("Image")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if config.activities:
        plt.subplot(2, 5, 2)
        plt.imshow(img[0, :, :, 4:5].numpy().squeeze(), cmap='gray')  # plot the "act_sitt" activity
        plt.title("Act_sitt")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(2, 5, 3)
        plt.imshow(img[0, :, :, 5:6].numpy().squeeze(), cmap='gray')  # plot the "act_lay" activity
        plt.title("Act_lay")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(2, 5, 4)

    plt.imshow(bound[0].numpy())
    plt.title("Boundary")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 5, 5)
    plt.imshow(room[0].numpy())
    plt.title("Room")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if config.furniture:
        plt.subplot(2, 5, 6)
        plt.imshow(furn[0].numpy())
        plt.title("Furniture")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    plt.subplot(2, 5, 7)
    plt.imshow(convert_one_hot_to_image(logcw)[0].numpy().squeeze())
    plt.title("Log CW")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 5, 8)
    plt.imshow(convert_one_hot_to_image(logr)[0].numpy().squeeze())
    plt.title("Log R")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if config.furniture:
        plt.subplot(2, 5, 9)
        plt.imshow(convert_one_hot_to_image(logf)[0].numpy().squeeze())
        plt.title("Log F")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    return figure

@tf.function
def test_step(
    furn_bool: bool,
    model: tf.keras.Model,
    optim: tf.keras.optimizers.Optimizer,
    img: tf.Tensor,
    hr: tf.Tensor,
    hb: tf.Tensor,
    hf: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # forward
    logits_r, logits_cw, logits_f = model(img)
    loss1 = balanced_entropy(logits_r, hr)
    loss2 = balanced_entropy(logits_cw, hb)
    
    if furn_bool:
        loss3 = balanced_entropy(logits_f, hf)
        w1, w2, w3 = cross_three_tasks_weight(hr, hb, hf)
        loss = w1 * loss1 + w2 * loss2 + w3 * loss3
    else:
        w1, w2, w3 = cross_two_tasks_weight(hr, hb)
        loss = w1 * loss1 + w2 * loss2
        loss3 = None
    return logits_r, logits_cw, logits_f, loss, loss1, loss2, loss3


def test(config: argparse.Namespace):
    # initialization
    writer = tf.summary.create_file_writer(config.logdir)
    pltiter = 0
    dataset, model, _ = init(config)
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    num_batches = 0
    furn_bool = config.furniture

    # testing loop
    for data in tqdm(list(dataset.batch(config.batchsize))):
        img, bound, room, furn, act_door, act_sitt, act_lay, act_wash, hb, hr, hf = preprocess(
            data['image'], data['boundary'], data['room'], data['furn'], data['act_door'], data['act_sitt'], data['act_lay'], data['act_wash']
        )
        logits_r, logits_cw, logits_f, loss, loss1, loss2, loss3 = test_step(furn_bool, model, img, hr, hb, hf
        )

        f = image_grid(config, img, bound, room, furn, logits_r, logits_cw, logits_f)
        im = plot_to_image(f)
        with writer.as_default():
            tf.summary.scalar("Loss", loss.numpy(), step=pltiter)
            tf.summary.scalar("LossR", loss1.numpy(), step=pltiter)
            tf.summary.scalar("LossB", loss2.numpy(), step=pltiter)
            if config.furniture:
                tf.summary.scalar("LossF", loss3.numpy(), step=pltiter)
            tf.summary.image("Data", im, step=pltiter)
        writer.flush()
        pltiter += 1

        # accumulate loss
        total_loss += loss
        total_loss1 += loss1
        total_loss2 += loss2
        total_loss3 += loss3
        num_batches += 1

    # calculate average loss over all batches
    avg_loss = total_loss / num_batches
    avg_loss1 = total_loss1 / num_batches
    avg_loss2 = total_loss2 / num_batches
    avg_loss3 = total_loss3 / num_batches

    print(f'Avg Loss: {avg_loss.numpy()}, Avg LossR: {avg_loss1.numpy()}, Avg LossB: {avg_loss2.numpy()}, Avg LossF: {avg_loss3.numpy()}')


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tfmodel", type=str, default="subclass", choices=["subclass", "func"]
    )
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--logdir", type=str, default="/local/home/amohap/data/tf2deep/test_furn_context/log/store")
    p.add_argument("--modeldir", type=str, default="/local/home/amohap/data/tf2deep/test_furn_context/model/store")
    p.add_argument("--datadir", type=str, default="/local/home/amohap/data/tf2deep")
    p.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    p.add_argument("--weight", type=str, default="/local/home/amohap/data/tf2deep/furniture_context/log/store/G")
    p.add_argument("--activities", action='store_true')
    p.add_argument("--furniture", action='store_true')
    p.add_argument("--save-tensor-interval", type=int, default=10)
    p.add_argument("--save-model-interval", type=int, default=20)
    p.add_argument("--tomlfile", type=str, default=None)
    p.add_argument(
        "--feature-channels",
        type=int,
        action="store",
        default=[256, 128, 64, 32],
        nargs=4,
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=["vgg16", "resnet50", "mobilenetv1", "mobilenetv2"],
    )
    p.add_argument(
        "--feature-names",
        type=str,
        action="store",
        nargs=5,
        default=[
            "block1_pool",
            "block2_pool",
            "block3_pool",
            "block4_pool",
            "block5_pool",
        ],
    )
    return p.parse_args(args)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = overwrite_args_with_toml(args)
    print(args)
    test(args)
