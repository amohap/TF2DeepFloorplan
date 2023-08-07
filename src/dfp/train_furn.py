import argparse
import io
import os
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
from .utils.rgb_ind_convertor import (floorplan_boundary_map,
                                      floorplan_furn_map, floorplan_room_map,
                                      ind2rgb)



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
        plt.imshow(img[0, :, :, 3:4].numpy().squeeze(), cmap='gray')  # plot the "act_sitt" activity
        plt.title("Act_sitt")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(2, 5, 3)
        plt.imshow(img[0, :, :, 4:5].numpy().squeeze(), cmap='gray')  # plot the "act_lay" activity
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


def image_grid_processed(
    config: argparse.Namespace,
    img: tf.Tensor,
    bound: tf.Tensor,
    room: tf.Tensor,
    furn: tf.Tensor,
    logr: tf.Tensor,
    logcw: tf.Tensor,
    logf: tf.Tensor
) -> matplotlib.figure.Figure:
    figure = plt.figure(figsize=(25,20))
    plt.subplot(3, 4, 1)
    plt.imshow(img[0, :, :, :3].numpy()) 
    plt.title("Input Image")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    plt.subplot(3, 4, 2)
    bound_ind = ind2rgb(bound[0].numpy(), floorplan_boundary_map)
    bound_plt = bound_ind.astype('uint8')
    plt.imshow(bound_plt)
    plt.title("Boundary Ground Truth")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    plt.subplot(3, 4, 3)
    room_ind = ind2rgb(room[0].numpy(), floorplan_room_map)
    room_plot = room_ind.astype('uint8')
    plt.imshow(room_plot)
    plt.title("Room Ground Truth")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if config.furniture:
        plt.subplot(3, 4, 4)
        furn_ind = ind2rgb(furn[0].numpy(), floorplan_furn_map)
        furn_plot = furn_ind.astype('uint8')
        plt.imshow(furn_plot)
        plt.title("Furniture Ground Truth")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    plt.subplot(3, 4, 6)
    cw_onehot = convert_one_hot_to_image(logcw)[0].numpy().squeeze()
    cw_ind = ind2rgb(cw_onehot, floorplan_boundary_map)
    cw_plot = cw_ind.astype('uint8')
    plt.imshow(cw_plot)
    plt.title("Boundary Prediction")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.subplot(3, 4, 7)
    logr_onehot = convert_one_hot_to_image(logr)[0].numpy().squeeze()
    logr_ind = ind2rgb(logr_onehot, floorplan_room_map)
    logr_plot = logr_ind.astype('uint8')
    plt.imshow(logr_plot)
    plt.title("Room Prediction")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if config.furniture:
        plt.subplot(3, 4, 8)
        logf_onehot = convert_one_hot_to_image(logf)[0].numpy().squeeze()
        logf_ind = ind2rgb(logf_onehot, floorplan_furn_map)
        logf_plot = logf_ind.astype('uint8')
        plt.imshow(logf_plot)
        plt.title("Furniture Prediction")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    
    if config.activities:
        plt.subplot(3, 4, 5)
        plt.imshow(img[0, :, :, 3:4].numpy().squeeze(), cmap='gray')
        plt.title("Act_sitt")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(3, 4, 9)
        plt.imshow(img[0, :, :, 4:5].numpy().squeeze(), cmap='gray') 
        plt.title("Act_lay")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    return figure


@tf.function
def train_step(
    furn_bool: bool,
    model: tf.keras.Model,
    optim: tf.keras.optimizers.Optimizer,
    img: tf.Tensor,
    hr: tf.Tensor,
    hb: tf.Tensor,
    hf: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # forward
    with tf.GradientTape() as tape:
    
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

    # backward
    grads = tape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return logits_r, logits_cw, logits_f, loss, loss1, loss2, loss3


def main(config: argparse.Namespace):
    # initialization
    writer = tf.summary.create_file_writer(config.logdir)
    pltiter = 0
    dataset, model, optim = init(config)
    furn_bool = config.furniture
  
    # training loop
    for epoch in range(config.epochs):
        print("[INFO] Epoch {}".format(epoch))
        for data in tqdm(list(dataset.shuffle(400).batch(config.batchsize))):
            img, bound, room, furn, act_door, act_sitt, act_lay, act_wash, hb, hr, hf = preprocess(
                data['image'], data['boundary'], data['room'], data['furn'], data['act_door'], data['act_sitt'], data['act_lay'], data['act_wash']
                )
            logits_r, logits_cw, logits_f, loss, loss1, loss2, loss3 = train_step(furn_bool, model, optim, img, hr, hb, hf)

            # plot progress
            if pltiter % config.save_tensor_interval == 0:
                f = image_grid_processed(config, img, bound, room, furn, logits_r, logits_cw, logits_f)
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

        # save model
        if epoch % config.save_model_interval == 0:
            model.save_weights(config.logdir + "/G")
            model.save(config.modeldir)
            print("[INFO] Saving Model ...")


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tfmodel", type=str, default="subclass", choices=["subclass", "func"]
    )
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--logdir", type=str, default="/local/home/amohap/data/tf2deep/furn_act_nocontext_100ep/log/store")
    p.add_argument("--modeldir", type=str, default="/local/home/amohap/data/tf2deep/furn_act_nocontext_100ep/model/store")
    p.add_argument("--datadir", type=str, default="/local/home/amohap/data/tf2deep")
    p.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    p.add_argument("--weight", type=str)
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
    main(args)
