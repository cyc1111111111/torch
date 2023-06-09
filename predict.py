# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2021/12/14 16:15
# @Software: PyCharm
# @Brief:


import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataloader import data_transform
from utils import create_model, model_parallel
from config import args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    image_path = "dataset/SAT/airplane.jpg"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)

    image = Image.open(image_path)
    plt.imshow(image)

    # [N, C, H, W]
    image = data_transform["val"](image)
    # expand batch dimension
    image = torch.unsqueeze(image, dim=0)

    # create model
    model = create_model(args)
    model = model_parallel(args, model).to(device)
    # load model weights
    model_weight_path = "{}/weights/epoch=49_val_acc=0.8743.pth".format(args.summary_dir)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        index = torch.argmax(predict).numpy()

    pre_res = "prediction: {}   prob: {:.3}\n".format(args.label_name[index],
                                                      predict[index].numpy())
    plt.title(pre_res)

    print("prediction: {}   prob: {:.3}\n".format(args.label_name[index],
                                                  predict[index].numpy()))
    for i in range(len(predict)):
        print("class: {}   prob: {:.3}".format(args.label_name[i],
                                               predict[i].numpy()))

    plt.show()


if __name__ == '__main__':
    main()
