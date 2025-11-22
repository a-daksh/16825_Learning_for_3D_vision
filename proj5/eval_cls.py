import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--theta', type=int, default=0, help='Rotates gt point cloud by theta in deg')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.output_dir = args.output_dir + "/cls"
    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    if args.theta:
        theta = torch.tensor(torch.pi*args.theta/180)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.eye(3)
        R[1, 1] = c
        R[1, 2] = -s
        R[2, 1] = s
        R[2, 2] = c
        test_data = test_data @ R.T

    num_samples=test_data.shape[0]
    correct=0
    total=0
    with torch.no_grad():
        for start in range(0, num_samples, args.batch_size):
            end = min(start + args.batch_size, num_samples)
            
            batch_data = test_data[start:end].to(args.device)
            batch_label = test_label[start:end].to(args.device)
            pred_label = model(batch_data).argmax(-1)

            correct += pred_label.eq(batch_label).sum().item()
            total += batch_label.size(0)

    # Compute Accuracy
    test_accuracy = correct / total
    print ("test accuracy: {}".format(test_accuracy))

    with torch.no_grad():
        sample_data = test_data[args.i:args.i+1].to(args.device)
        pred_label_sample = model(sample_data).argmax(-1).cpu()[0].item()
    
    viz_cls(test_data[args.i], test_label[args.i].item(), "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_cls(test_data[args.i], pred_label_sample, "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
