import os
import torch
from torchvision import transforms, datasets
from augmentation import Aug
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
import pickle
from model.network import Network
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--dataset", type=str, default='./dataset/FF++',
                        help="Dataset root directory.")
    parser.add_argument("-c", "--checkpoints", type=str, default='./checkpoints',
                        help="The checkpoints.")
    parser.add_argument("-j", "--loss_folder", type=str, default='./results',
                        help='The folder to save the loss values.')
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help='Batch size.')
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument("-l", "--learning_rate", type=int, default=1e-4,
                        help='The learning_rate.')
    parser.add_argument("-y", "--weight_decay", type=int, default=1e-4,
                        help='The weight_decay.')
    parser.add_argument("-i", "--image_size", type=int, default=224,
                        help='The size of the input image.')
    parser.add_argument("-k", "--patch_size", type=int, default=3,
                        help='Kernel size for conv layer for feature extraction.')
    parser.add_argument("-s", "--stride", type=int, default=1,
                        help='Stride size for conv layer for feature extraction.')
    parser.add_argument("-d", "--base_dims", type=list, default=[128, 128, 128],
                        help='Dimensions of each attention head at each stage of the transformer.')
    parser.add_argument("-w", "--depth", type=list, default=[8, 8, 8],
                        help='The number of transformer_blocks in each stage of transformer.')
    parser.add_argument("-z", "--heads", type=list, default=[4, 8, 16],
                        help='Number of attention heads at each stage of the transformer.')
    parser.add_argument("-m", "--mlp_ratio", type=int, default=4,
                        help='The FeedForward layer expands the number of neurons in the input layer by times.')
    parser.add_argument("-n", "--num_classes", type=int, default=2,
                        help='The number of classification categories.')
    parser.add_argument("-f", "--in_chans", type=int, default=512,
                        help='The num of channels for input of this network.')
    parser.add_argument("-o", "--attn_drop", type=int, default=0,
                        help='Dropout attention map to prevent overfitting.')
    parser.add_argument("-p", "--proj_drop", type=int, default=0,
                        help='Dropout fully connected layer to prevent overfitting.')

    args = parser.parse_args()
    return args


def MyDataload(args):
    # ImageNet dataset mean and std
    mean = [0.4718, 0.3467, 0.3154]
    std = [0.1656, 0.1432, 0.1364]

    # Data preprocessing
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        Aug(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(root=args.dataset + '/train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=args.dataset + '/val', transform=val_transforms)

    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=RandomSampler(val_dataset),
        batch_size=batch_size
    )
    print("Finished Constructing Dataloader. ")

    return train_dataloader, val_dataloader, train_dataset, val_dataset


def train(model, train_dataloader, criterion, optimizer, epoch, epochs, device, batch_size, train_dataset, scheduler,
          train_loss, train_accu):
    m = nn.Softmax(dim=1)
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    running_train_loss = 0.0
    running_train_corrects = 0
    train_phase_idx = 0

    # Train
    model.train()
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(inputs)
            _, preds = torch.max(m(output), 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        if train_phase_idx % 100 == 0:
            print('Train loss:', train_phase_idx, ':', loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, train_phase_idx * batch_size, len(train_dataset),
                       100. * train_phase_idx * batch_size / len(train_dataset),
                loss.item()))
        train_phase_idx += 1

        running_train_loss += loss.item() * inputs.size(0)
        running_train_corrects += torch.sum(preds == labels.data)

    scheduler.step()

    epoch_loss = running_train_loss / len(train_dataset)
    epoch_acc = running_train_corrects.double() / len(train_dataset)
    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def val(model, epoch, n, val_dataloader, device, optimizer, criterion, batch_size, val_dataset, val_loss, val_accu,
        time_elapsed_val, min_loss, best_model_wts):
    running_val_loss = 0.0
    running_val_corrects = 0
    val_phase_idx = 0
    m = nn.Softmax(dim=1)

    # Validation
    model.eval()
    since_val = time.time()
    for inputs_val, labels_val in val_dataloader:
        inputs_val = inputs_val.to(device)
        labels_val = labels_val.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            output_val = model(inputs_val)
            _, preds_val = torch.max(m(output_val), 1)
            loss_val = criterion(output_val, labels_val)

        if val_phase_idx % 100 == 0:
            print('Validation loss:', val_phase_idx, ':', loss_val.item())
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, val_phase_idx * batch_size, len(val_dataset),
                       100. * val_phase_idx * batch_size / len(val_dataset),
                loss_val.item()))
        val_phase_idx += 1

        running_val_loss += loss_val.item() * inputs_val.size(0)
        running_val_corrects += torch.sum(preds_val == labels_val.data)

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_acc = running_val_corrects.double() / len(val_dataset)
    val_loss.append(epoch_val_loss)
    val_accu.append(epoch_val_acc)
    time_elapsed_val += time.time() - since_val
    print('Validation Loss: {:.5f} Acc: {:.5f}'.format(epoch_val_loss, epoch_val_acc))

    if epoch_val_loss < min_loss:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_val_loss, min_loss))
        min_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    if not os.path.isdir(args.checkpoints):
        os.mkdir(args.checkpoints)

    state = {
        'state_dict': model.state_dict(),
        'epoch': epoch
    }
    torch.save(state, args.checkpoints + '/model_epoch' + str(n))
    n += 1
    print('Total val  complete in {:.0f}m {:.0f}s'.format(time_elapsed_val // 60, time_elapsed_val % 60))

    return best_model_wts, min_loss


def save(train_loss, train_accu, val_loss, val_accu, best_model_wts, min_loss, time_elapsed, time_elapsed_val):
    if not os.path.isdir(args.loss_folder):
        os.mkdir(args.loss_folder)
    with open(args.loss_folder + '/model_loss.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)
    if not os.path.isdir(args.checkpoints):
        os.mkdir(args.checkpoints)

    state = {'state_dict': best_model_wts,
             'min_loss': min_loss,
             'total train and val time': time_elapsed // 60,
             'total val time': time_elapsed_val // 60}
    torch.save(state, args.checkpoints + '/model_best.pth')


def main():
    train_dataloader, val_dataloader, train_dataset, val_dataset = MyDataload(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(args.image_size, args.patch_size, args.stride, args.base_dims, args.depth, args.heads,
                    args.mlp_ratio, args.num_classes, args.in_chans, args.attn_drop, args.proj_drop)

    model.to(device)
    print(" model Loaded..")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed_val = 0
    min_loss = 10000
    n = 0

    for epoch in range(args.epochs):
        train(model, train_dataloader, criterion, optimizer, epoch, args.epochs, device, args.batch_size, train_dataset,
              scheduler, train_loss, train_accu)
        best_model_wts, min_loss = val(model, args.epochs, n, val_dataloader, device, optimizer, criterion,
                                       args.batch_size, val_dataset, val_loss, val_accu, time_elapsed_val, min_loss,
                                       best_model_wts)

    time_elapsed = time.time() - since
    print('Training and Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    save(train_loss, train_accu, val_loss, val_accu, best_model_wts, min_loss, time_elapsed, time_elapsed_val)


if __name__ == '__main__':
    args = parse_args()
    main()
