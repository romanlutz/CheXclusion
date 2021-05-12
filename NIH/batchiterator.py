
import torch
from utils import *
import numpy as np

def BatchIterator(model, phase,
        Data_loader,
        criterion,
        optimizer,
        device):

    # --------------------  Initial parameters
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 1000
    running_loss = 0.0

    print("BatchIterator starting to read images now.")
    for i, data in enumerate(Data_loader):
        print("Reading batch " + str(i))
        imgs, labels, _ = data

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            print("Starting model training on batch " + str(i))
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(imgs)

        loss = criterion(outputs, labels)
        print("Loss: " + str(loss))

        if phase == 'train':

            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights
            print("Finished model training on batch " + str(i))

        running_loss += loss * batch_size
        if (i % 10 == 0):
            print("Read " + str(i * batch_size) + " images.")

    return running_loss
