import argparse
import io
import os
import re
import sys
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .data_furn import convert_one_hot_to_image, loadDataset, preprocess
from .loss import (balanced_entropy, cross_three_tasks_weight,
                   cross_two_tasks_weight)
from .net import deepfloorplanModel, deepfurnfloorplanModel
from .net_func import deepfloorplanFunc
from .utils.rgb_ind_convertor import (floorplan_boundary_map,
                                      floorplan_furn_map, floorplan_room_map,
                                      ind2rgb)
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
    optim = tf.keras.optimizers.Adam(learning_rate=config.lr)
    return dataset, model, optim


def extract_numbers_from_file(filename):
   
    with open(filename, 'r') as f:
        lines = f.readlines()

    regex = re.compile(r'(\d+)\.png')
    numbers = []

    for line in lines:
        match = regex.search(line)

        if match:
            number = match.group(1)
            numbers.append(int(number))

    return numbers


def plot_to_image(figure: matplotlib.figure.Figure) -> tf.Tensor:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


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
        plt.title("Sitting")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplot(3, 4, 9)
        plt.imshow(img[0, :, :, 4:5].numpy().squeeze(), cmap='gray') 
        plt.title("Laying")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    return figure

@tf.function
def test_step(
    furn_bool: bool,
    model: tf.keras.Model,
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
    save_img_bool = config.saveimg
    
    if save_img_bool:
        txt_filename = config.txtdir
        os.makedirs(config.saveimgdir, exist_ok=True)
        os.makedirs(os.path.join(config.saveimgdir, "boundary"))
        os.makedirs(os.path.join(config.saveimgdir, "boundary", "GT"))
        os.makedirs(os.path.join(config.saveimgdir, "boundary", "predictions"))
        os.makedirs(os.path.join(config.saveimgdir, "room"))
        os.makedirs(os.path.join(config.saveimgdir, "room", "GT"))
        os.makedirs(os.path.join(config.saveimgdir, "room", "predictions"))
        
        if furn_bool:
            os.makedirs(os.path.join(config.saveimgdir, "furniture"))
            os.makedirs(os.path.join(config.saveimgdir, "furniture", "GT"))
            os.makedirs(os.path.join(config.saveimgdir, "furniture", "predictions"))
        
        scenes = extract_numbers_from_file(txt_filename)
    
    else:
        scenes = range(len(dataset))

    # testing loop
    for (data, scene) in tqdm(zip(list(dataset.batch(config.batchsize)), scenes)):
        img, bound, room, furn, _, _, _, _, hb, hr, hf = preprocess(
            data['image'], data['boundary'], data['room'], data['furn'], data['act_door'], data['act_sitt'], data['act_lay'], data['act_wash']
        )
        logits_r, logits_cw, logits_f, loss, loss1, loss2, loss3 = test_step(furn_bool, model, img, hr, hb, hf
        )

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

        if save_img_bool:
            filename = f"{scene:05}.png" 

            save_path = os.path.join(config.saveimgdir, "boundary", "GT", filename)
            bound_ind = ind2rgb(bound[0].numpy(), floorplan_boundary_map)
            bound_plt = bound_ind.astype('uint8')
            plt.imshow(bound_plt)
            plt.axis('off')            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            save_path = os.path.join(config.saveimgdir, "boundary", "predictions", filename)
            cw_onehot = convert_one_hot_to_image(logits_cw)[0].numpy().squeeze()
            cw_ind = ind2rgb(cw_onehot, floorplan_boundary_map)
            cw_plot = cw_ind.astype('uint8')
            plt.imshow(cw_plot)
            plt.axis('off')            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            save_path = os.path.join(config.saveimgdir, "room", "GT", filename)
            room_ind = ind2rgb(room[0].numpy(), floorplan_room_map)
            room_plt = room_ind.astype('uint8')
            plt.imshow(room_plt)
            plt.axis('off')            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            save_path = os.path.join(config.saveimgdir, "room", "predictions", filename)
            r_onehot = convert_one_hot_to_image(logits_r)[0].numpy().squeeze()
            r_ind = ind2rgb(r_onehot, floorplan_room_map)
            r_plot = r_ind.astype('uint8')
            plt.imshow(r_plot)
            plt.axis('off')            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            save_path = os.path.join(config.saveimgdir, "furniture", "GT", filename)
            furn_ind = ind2rgb(furn[0].numpy(), floorplan_furn_map)
            furn_plt = furn_ind.astype('uint8')
            plt.imshow(furn_plt)
            plt.axis('off')            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            save_path = os.path.join(config.saveimgdir, "furniture", "predictions", filename)
            f_onehot = convert_one_hot_to_image(logits_f)[0].numpy().squeeze()
            f_ind = ind2rgb(f_onehot, floorplan_room_map)
            f_plot = f_ind.astype('uint8')
            plt.imshow(f_plot)
            plt.axis('off')            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

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
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--logdir", type=str, default="log/store")
    p.add_argument("--modeldir", type=str, default="model/store")
    p.add_argument("--datadir", type=str, default="data/tf2deep")
    p.add_argument("--txtdir", type=str, default="data/tf2deep/tf2deep_activities_furn_test.txt")
    p.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    p.add_argument("--weight", type=str, default="log/store/G")
    p.add_argument("--activities", action='store_true')
    p.add_argument("--furniture", action='store_true')
    p.add_argument("--saveimg", action='store_true')
    p.add_argument("--saveimgdir", type=str, default="images/")
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
