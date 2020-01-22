import argparse
import torch


def init_train():
    parser_train = argparse.ArgumentParser(description='CLI for the image classification project.')
    parser_train.add_argument('data_directory', type=str, help='Assign a directory where the training data is stored. When running from root, use ./input/flowers')
    parser_train.add_argument('--save_directory', type=str, help='Assign a directory where to save the model after it was trained')
    parser_train.add_argument('--arch', default='densenet121', type=str, help='Assign the architecture for the transfer learning model')
    parser_train.add_argument('--learning_rate', default=0.003, type=int, help='Assign the learning rate you want to use during the training phase')
    parser_train.add_argument('--hidden_units', default=256, type=int, help='Assign how many hidden units you want to have in your architecture')
    parser_train.add_argument('--epochs', default=10, type=int, help='Assign the amount of epochs for the training phase')
    parser_train.add_argument('--gpu', action='store_true', help='Force to run in on the GPU always')

    args_train = parser_train.parse_args()
    data_dir = args_train.data_directory
    save_dir = args_train.save_directory
    arch = args_train.arch
    learning_rate = args_train.learning_rate
    hidden_units = args_train.hidden_units
    epochs = args_train.epochs
    gpu = args_train.gpu
    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return data_dir, save_dir, arch, hidden_units, learning_rate, epochs, device



def init_predict():
    parser_predict = argparse.ArgumentParser(description='CLI for the image classification project')
    parser_predict.add_argument('picture_path', type=str, help='Assign the path to the picture you want to predict')
    parser_predict.add_argument('checkpoint', type=str, help='Assign a checkpoint name you want to use for prediction')
    parser_predict.add_argument('--top_k', type=int, default=5, help='Assign how many classes you want to see as the output')
    parser_predict.add_argument('--category_names', type=str, help='Assign a file name that contains the category mappings')
    parser_predict.add_argument('--gpu', action='store_true', help='Force to run in on the GPU always')

    args_predict = parser_predict.parse_args()
    picture_path = args_predict.picture_path
    checkpoint = args_predict.checkpoint
    top_k = args_predict.top_k
    category_file = args_predict.category_names
    gpu = args_predict.gpu
    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return picture_path, checkpoint, top_k, category_file, device
