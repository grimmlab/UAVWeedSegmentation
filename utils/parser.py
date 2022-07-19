import argparse

def create_train_parser():
    my_parser = argparse.ArgumentParser(
        description='Trains deep learning models for Weed/Crop Segmentation using kFold Cross validation and saves the results in a optuna database')

    my_parser.add_argument('--db_name',
                        metavar='db_name',
                        type=str,
                        help='Name of the optuna database', default="")

    my_parser.add_argument('--study_name',
                        metavar='study_name',
                        type=str,
                        help='Name of the optuna study', default="")

    my_parser.add_argument('architecture',
                        type=str,
                        help='String of an architecture, implemented: fcn8s, fcn16s, fcn32s, unet, dlplus')

    my_parser.add_argument('encoder_name',
                        type=str,
                        help='String of an encoder (feature extractor), implemented: resnet18, resnet34, resnet50, resnet101')
    
    my_parser.add_argument('--run_prefix',
                        type=str,
                        help='Prefix for the optuna database', default="db")
    
    my_parser.add_argument('--save_checkpoint',
                        help='Bool, if True will save checkpoints of each epoch.',
                        action='store_true')

    my_parser.add_argument('--pretrained',
                        help='Bool, if True will use a pretrained feature extractor (on ImageNet)',
                        action='store_true', default=True)
    
    my_parser.add_argument('--b_bilinear',
                        help='Bool, if True will use bilinear interpolation, if False will use transposed convolutions',
                        action='store_true', default=True)

    my_parser.add_argument('--replace_stride_with_dilation',
                        help='Bool, if True will replace strides with dilated convolutions',
                        action='store_true')

    my_parser.add_argument('--b_clean_study',
                        help='Bool, if True will delete all Trials and start a fresh study',
                        action='store_true')

    my_parser.add_argument('--lr_max',
                        type=float,
                        help='Maximal learning rate to sample from', default=1e-2)

    my_parser.add_argument('--lr_min',
                        type=float,
                        help='Minimal learning rate to sample from', default=1e-4)

    my_parser.add_argument('--n_folds',
                        type=int,
                        help='Number of folds for kFold Cross validation', default=4)

    my_parser.add_argument('--batch_size',
                        type=int,
                        help='Number patches per batch', default=100)

    my_parser.add_argument('--n_trials',
                        type=int,
                        help='Number of trials per optuna study', default=50)

    my_parser.add_argument('--seed',
                        type=int,
                        help='Seed of the experiment', default=42)

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")
    
    args = my_parser.parse_args()
    return args

def create_test_parser():
    my_parser = argparse.ArgumentParser(
        description='Predicts segmentation masks on complete UAV captures')

    my_parser.add_argument('model',
                        metavar='model',
                        type=str,
                        help='Path to retrained model.pt file that is used to generate predictions.')

    my_parser.add_argument('subset',
                        metavar='subset',
                        type=str,
                        help='String of the subset folder in /data to predict images on.')

    my_parser.add_argument('--batch_size',
                        type=int,
                        help='Number patches per batch', default=20)

    my_parser.add_argument('--seed',
                        type=int,
                        help='Seed of the experiment', default=42)

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")
    
    args = my_parser.parse_args()
    return args

def compare_studies_parser():
    my_parser = argparse.ArgumentParser(
        description='Compares different study databases in /results')

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")
    
    args = my_parser.parse_args()
    return args


def generate_patches_parser():
    my_parser = argparse.ArgumentParser(
        description='Generates patches from UAV captures and their masks')

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")
    
    args = my_parser.parse_args()
    return args


def create_compare_parser():
    my_parser = argparse.ArgumentParser(
        description='Compares Ground Truth with Predictions')

    my_parser.add_argument('--bbch',
                        type=str,
                        help='BBCH stage as string', default=None)
    
    my_parser.add_argument('subset',
                        metavar='subset',
                        type=str,
                        help='String of the subset folder in /data to predict images on.')

    args = my_parser.parse_args()
    return args