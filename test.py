import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler
import sklearn.metrics as metrics
from sklearn.metrics import auc
import pickle
import argparse
from model.network import Network


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--dataset", type=str, default='./dataset/',
                        help="The root directory of dataset.")
    parser.add_argument("-c", "--checkpoints", type=str, default='./checkpoints/model_best.pth',
                        help="The checkpoints of network.")
    parser.add_argument("-j", "--results", type=str, default='./results',
                        help='The folder to store the results.')
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help='Batch size.')
    parser.add_argument("-i", "--image_size", type=int, default=224,
                        help='The size of the input image.')
    parser.add_argument("-k", "--patch_size", type=int, default=3,
                        help='Kernel size for conv layer for feature extraction.')
    parser.add_argument("-s", "--stride", type=int, default=1,
                        help='Stride size for conv layer for feature extraction.')
    parser.add_argument("-d", "--base_dims", type=list, default=[128, 128, 128],
                        help='Dimensions of each attention head at each stage of the transformer.')
    parser.add_argument("-w", "--depth", type=list, default=[8, 8, 8],
                        help='The number of transformer blocks in each stage of transformer.')
    parser.add_argument("-z", "--heads", type=list, default=[4, 8, 16],
                        help='Number of attention heads at each stage of the transformer.')
    parser.add_argument("-m", "--mlp_ratio", type=int, default=4,
                        help='The FeedForward layer expands the number of neurons in the input layer by times.')
    parser.add_argument("-n", "--num_classes", type=int, default=2,
                        help='The number of classification categories.')
    parser.add_argument("-f", "--in_chans", type=int, default=512,
                        help='The num of channels for input of this network.')
    parser.add_argument("-o", "--attn_drop", type=int, default=0,
                        help='dropout attention map to prevent overfitting.')
    parser.add_argument("-p", "--proj_drop", type=int, default=0,
                        help='dropout fully connected layer to prevent overfitting.')

    args = parser.parse_args()
    return args


def my_test_dataloder(dataset, args):
    # ImageNet datasets mean and std
    mean = [0.4718, 0.3467, 0.3154]
    std = [0.1656, 0.1432, 0.1364]

    # Data preprocessing
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = datasets.ImageFolder(root=args.dataset + dataset, transform=test_transforms)
    test_dataset_size = len(test_dataset)
    batch_size = args.batch_size
    test_dataloader = DataLoader(
        test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=batch_size
    )
    print("Finished constructing test dataloader!")

    return test_dataloader, test_dataset_size


def test(model, test_dataloader, device, dataset, test_dataset_size):
    model.eval()
    m = nn.Softmax(dim=1)
    Sum = 0
    y_real_label = []
    y_score = []

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs).to(device).float()
        output_score = m(output).to(device)
        output_scores = output_score.detach().cpu().numpy()
        y_scores = output_scores[:, 1]
        for i in y_scores:
            y_score.append(i)
        _, prediction = torch.max(output, 1)
        pred_label = prediction.detach().cpu().numpy()
        main_label = labels.detach().cpu().numpy()
        for i in main_label:
            y_real_label.append(i)
        bool_list = list(
            map(lambda x, y: x == y, pred_label, main_label))
        Sum += sum(np.array(
            bool_list) * 1)
    print(dataset + ' Prediction Acc: ', (Sum / test_dataset_size) * 100, '%')

    return y_real_label, y_score


def test_AUC(y_real_label, y_score, dataset):
    y_test = np.array(y_real_label)
    y_pros = np.array(y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pros, pos_label=1)
    roc_auc = auc(fpr, tpr)
    with open(args.results + '/model_auc.pkl', 'wb') as f:
        pickle.dump([fpr, tpr, roc_auc], f)
    print(dataset + ' AUC:%0.5f' % roc_auc)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(args.image_size, args.patch_size, args.stride, args.base_dims, args.depth, args.heads,
                    args.mlp_ratio, args.num_classes, args.in_chans, args.attn_drop, args.proj_drop)

    state = torch.load(args.checkpoints)
    checkpoint = state['state_dict']
    model.load_state_dict(checkpoint)

    model.to(device)
    print("Finished load best weights.")

    dataset_names = ['Celeb-DF/test', 'DFDC/test', 'FF++C23/test', 'DeeperForensics/test']
    for dataset_name in dataset_names:
        test_dataloader, test_dataset_size = my_test_dataloder(dataset_name, args)
        y_real_label, y_score = test(model, test_dataloader, device, dataset_name, test_dataset_size)
        test_AUC(y_real_label, y_score, dataset_name)


if __name__ == '__main__':
    args = parse_args()
    main()
