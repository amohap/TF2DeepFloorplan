import argparse
import gc
import os
import cv2
import sys
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from .data_furn import convert_one_hot_to_image
from .net import deepfloorplanModel, deepfurnfloorplanModel
from .net_func import deepfloorplanFunc
from .utils.rgb_ind_convertor import (
    floorplan_boundary_map,
    floorplan_fuse_map,
    floorplan_furn_map,
    ind2rgb,
)
from .utils.settings import overwrite_args_with_toml
from .utils.util import fill_break_line, flood_fill, refine_room_region

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def init(
    config: argparse.Namespace,
) -> Tuple[tf.keras.Model, tf.Tensor, np.ndarray]:
    if config.tfmodel == "subclass":
        #model = deepfloorplanModel(config=config)
        model = deepfurnfloorplanModel(config=config)
    elif config.tfmodel == "func":
        model = deepfloorplanFunc(config=config)
    if config.loadmethod == "log":
        model.load_weights(config.weight)
    elif config.loadmethod == "pb":
        model = tf.keras.models.load_model(config.weight)
    elif config.loadmethod == "tflite":
        model = tf.lite.Interpreter(model_path=config.weight)
        model.allocate_tensors()
    
    img = mpimg.imread(config.image)[:, :, :3]
    shp = img.shape

    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.image.resize(img, [512, 512])
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [-1, 512, 512, 3])
    if tf.math.reduce_max(img) > 1.0:
        img /= 255
    
    ## read image differently for real images
    if config.real:
        img = cv2.imread(config.image)
        shp = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (resize(img, (512, 512)) * 255).astype(np.uint8)
        img = tf.cast(img, dtype=tf.float32) / 255
        img = tf.reshape(img, [-1, 512, 512, 3])


    # load activities
    act_0 = cv2.imread(config.act_0, cv2.IMREAD_GRAYSCALE)
    act_0 = (resize(act_0, (512, 512)) * 255).astype(np.uint8) / 255.
    act_0 = (act_0>0.5).astype(np.uint8)
    act_0 = tf.convert_to_tensor(act_0, dtype=tf.uint8)
    act_0 = tf.cast(act_0, dtype=tf.float32)
    act_0 = tf.reshape(act_0, [-1, 512, 512, 1])

    act_1 = cv2.imread(config.act_1, cv2.IMREAD_GRAYSCALE)
    act_1 = (resize(act_1, (512, 512)) * 255).astype(np.uint8) / 255.
    act_1 = (act_1>0.5).astype(np.uint8)
    act_1 = tf.convert_to_tensor(act_1, dtype=tf.uint8)
    act_1 = tf.cast(act_1, dtype=tf.float32)
    act_1 = tf.reshape(act_1, [-1, 512, 512, 1])

    act_2 = cv2.imread(config.act_2, cv2.IMREAD_GRAYSCALE)
    act_2 = (resize(act_2, (512, 512)) * 255).astype(np.uint8) / 255.
    act_2 = (act_2>0.5).astype(np.uint8)
    act_2 = tf.convert_to_tensor(act_2, dtype=tf.uint8)
    act_2 = tf.cast(act_2, dtype=tf.float32)
    act_2 = tf.reshape(act_2, [-1, 512, 512, 1])

    act_3 = cv2.imread(config.act_3, cv2.IMREAD_GRAYSCALE)
    act_3 = (resize(act_3, (512, 512)) * 255).astype(np.uint8) / 255.
    act_3 = (act_3>0.5).astype(np.uint8)
    act_3 = tf.convert_to_tensor(act_3, dtype=tf.uint8)
    act_3 = tf.cast(act_3, dtype=tf.float32)
    act_3 = tf.reshape(act_3, [-1, 512, 512, 1])

    # concatenate result
    img = tf.concat([img, act_2, act_1, act_0, act_3], axis=-1)

    if config.loadmethod == "tflite":
        return model, img, shp
    model.trainable = False
    if config.tfmodel == "subclass":
        model.vgg16.trainable = False
    return model, img, shp


def predict(model: tf.keras.Model, img: tf.Tensor, shp: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    image = img[..., :3]
    activities = img[..., 3:]
    activities = model.activity_pipeline(activities)

    # VGG16 forward propagation
    features = []
    feature = image
    for layer in model.vgg16.layers:
        feature = layer(feature)
        if layer.name in model.feature_names:
            features.append(feature)
    
    # Room boundary pipeline
    features_rbp = []
    x_rbp = tf.identity(feature)
    features = features[::-1]
    for i in range(len(model.rbpups)):
        x_rbp = model.rbpups[i](x_rbp) + model.rbpcv1[i](features[i + 1])
        x_rbp = model.rbpcv2[i](x_rbp)
        features_rbp.append(x_rbp)
    logits_cw = tf.keras.backend.resize_images(model.rbpfinal(x_rbp), 2, 2, "channels_last")

    # Room type pipeline
    x_rt = tf.concat([tf.identity(feature), activities], axis=-1)
    for i in range(len(model.rtpups)):
        x_rt = model.rtpups[i](x_rt) + model.rtpcv1[i](features[i + 1])
        x_rt = model.rtpcv2[i](x_rt)
        x_rt = model.non_local_context(features_rbp[i], x_rt, i)
    logits_r = tf.keras.backend.resize_images(model.rtpfinal(x_rt), 2, 2, "channels_last")

    # Room furniture pipeline
    x_rf = tf.concat([tf.identity(feature), activities], axis=-1)
    for i in range(len(model.ftpups)):
        x_rf = model.ftpups[i](x_rf) + model.ftpcv1[i](features[i + 1])
        x_rf = model.ftpcv2[i](x_rf)
        x_rf = model.non_local_context(features_rbp[i], x_rf, i)
    logits_f = tf.keras.backend.resize_images(model.ftpfinal(x_rf), 2, 2, "channels_last")
        
    return logits_r, logits_cw, logits_f


def post_process(
    rm_ind: np.ndarray, bd_ind: np.ndarray, shp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hard_c = (bd_ind > 0).astype(np.uint8)
    # reg. from room prediction
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # reg. from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)
    cw_mask = np.reshape(cw_mask, (*shp[:2], -1))
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask.reshape(*shp[:2], -1) * new_rm_ind
    new_bd_ind = fill_break_line(bd_ind).squeeze()
    return new_rm_ind, new_bd_ind


def colorize(r: np.ndarray, cw: np.ndarray, f:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cr = ind2rgb(r, color_map=floorplan_fuse_map)
    ccw = ind2rgb(cw, color_map=floorplan_boundary_map)
    cf = ind2rgb(f, color_map=floorplan_furn_map)
    return cr, ccw, cf


def main(config: argparse.Namespace) -> np.ndarray:
    model, img, shp = init(config)
    if config.loadmethod == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], img)
        model.invoke()
        ri, cwi, f = 0, 1, 2
        if config.tfmodel == "func":
            ri, cwi = 1, 0
        logits_r = model.get_tensor(output_details[ri]["index"])
        logits_cw = model.get_tensor(output_details[cwi]["index"])
        logits_f = model.get_tensor(output_details[f]["index"])
        logits_cw = tf.convert_to_tensor(logits_cw)
        logits_r = tf.convert_to_tensor(logits_r)
        logits_f = tf.convert_to_tensor(logits_f)
    else:
        if config.tfmodel == "func":
            logits_r, logits_cw = model.predict(img)
        elif config.tfmodel == "subclass":
            if config.loadmethod == "log":
                logits_r, logits_cw, logits_f = predict(model, img, shp)
            elif config.loadmethod == "pb" or config.loadmethod == "none":
                logits_r, logits_cw = model(img)

    logits_r = tf.image.resize(logits_r, shp[:2])
    logits_cw = tf.image.resize(logits_cw, shp[:2])
    logits_f = tf.image.resize(logits_f, shp[:2])

    r = convert_one_hot_to_image(logits_r)[0].numpy()
    cw = convert_one_hot_to_image(logits_cw)[0].numpy()
    f = convert_one_hot_to_image(logits_f)[0].numpy()

    if not os.path.exists(config.plot_path):
        os.makedirs(config.plot_path)

    if not config.colorize and not config.postprocess:
        cw[cw == 1] = 9
        cw[cw == 2] = 10
        r[cw != 0] = 0
        result = (r + cw + f).squeeze()
        mpimg.imsave(os.path.join(config.plot_path, "boundary.png"), cw.squeeze())
        mpimg.imsave(os.path.join(config.plot_path, "room.png"), r.squeeze())
        mpimg.imsave(os.path.join(config.plot_path, "furn.png"), f.squeeze())
        mpimg.imsave(os.path.join(config.plot_path, "combined_result.png"), result.astype(np.uint8))
        return (r + cw).squeeze()
    
    elif config.colorize and not config.postprocess:
        r_color, cw_color, f_color = colorize(r.squeeze(), cw.squeeze(), f.squeeze())
        result = (r_color + cw_color + f_color)
        mpimg.imsave(os.path.join(config.plot_path,"boundary_col.png"), cw_color.astype(np.uint8))
        mpimg.imsave(os.path.join(config.plot_path,"room_col.png"), r_color.astype(np.uint8))
        mpimg.imsave(os.path.join(config.plot_path,"furn_col.png"), f_color.astype(np.uint8))
        mpimg.imsave(os.path.join(config.plot_path,"combined_result_col.png"), result.astype(np.uint8))
        return r_color + cw_color + f_color

    newr, newcw = post_process(r, cw, shp)
    if not config.colorize and config.postprocess:
        newcw[newcw == 1] = 9
        newcw[newcw == 2] = 10
        newr[newcw != 0] = 0
        result = newr.squeeze() + newcw.squeeze() + f.squeeze()
        mpimg.imsave(os.path.join(config.plot_path,"boundary_post.png"), newcw.squeeze().astype(np.uint8))
        mpimg.imsave(os.path.join(config.plot_path,"room_post.png"), newr.squeeze().astype(np.uint8))
        mpimg.imsave(os.path.join(config.plot_path,"furn_post.png"), f.squeeze().astype(np.uint8))
        mpimg.imsave(os.path.join(config.plot_path,"combined_result_post.png"), result.astype(np.uint8))
        return newr.squeeze() + newcw
    
    newr_color, newcw_color, new_f_color = colorize(newr.squeeze(), newcw.squeeze(), f.squeeze())
    result = newr_color + newcw_color + new_f_color
    mpimg.imsave(os.path.join(config.plot_path,"boundary_post_col.png"), newcw_color.astype(np.uint8))
    mpimg.imsave(os.path.join(config.plot_path,"room_post_col.png"), newr_color.astype(np.uint8))
    mpimg.imsave(os.path.join(config.plot_path,"furn_post_col.png"), new_f_color.astype(np.uint8))
    mpimg.imsave(os.path.join(config.plot_path,"combined_result_post_col.png"), result.astype(np.uint8))

    if config.save:
        mpimg.imsave(config.save, result.astype(np.uint8))

    return result


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tfmodel", type=str, default="subclass", choices=["subclass", "func"]
    )
    p.add_argument("--image", type=str, default="/local/home/amohap/data/pix2pix/results/tf2deep_230623/test_70_5to10traj/images/00009_9_fake_B.png")
    p.add_argument("--act_0", type=str, default="/local/home/amohap/data/pix2pix/results/tf2deep_230623/00009_act_laying.png")
    p.add_argument("--act_1", type=str, default="/local/home/amohap/data/pix2pix/results/tf2deep_230623/00009_act_sitting.png")
    p.add_argument("--act_2", type=str, default="/local/home/amohap/data/pix2pix/results/tf2deep_230623/00009_act_opening_door.png")
    p.add_argument("--act_3", type=str, default="/local/home/amohap/data/pix2pix/results/tf2deep_230623/00009_act_washing_hands.png")
    #p.add_argument("--weight", type=str, default="/local/home/amohap/data/tf2deep/furn_act_context/model/store/model.tflite")
    p.add_argument("--plot_path", type=str, default="dfp/plots/predicted_picture_00009_9")
    p.add_argument("--weight", type=str, default="/local/home/amohap/data/tf2deep/furn_act_context/log/store/G")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--colorize", action="store_true")
    p.add_argument("--real", action="store_true")
    p.add_argument(
        "--loadmethod",
        type=str,
        default="log",
        choices=["log", "tflite", "pb", "none"],
    )  # log,tflite,pb
    p.add_argument("--save", type=str)
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
    p.add_argument("--tomlfile", type=str, default=None)
    return p.parse_args(args)


def deploy_plot_res(result: np.ndarray):
    print(result.shape)
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = overwrite_args_with_toml(args)
    result = main(args)
    deploy_plot_res(result)
    plt.show()
