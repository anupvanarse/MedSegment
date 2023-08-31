import datetime
import os
import argparse
import tensorflow as tf  # conda install -c anaconda tensorflow
import settings   # Use the custom settings.py file for default parameters
from utils.dataloader import DatasetGenerator, get_decathlon_filelist
import numpy as np

DATA_PATH=os.path.join("/mnt/datasets/medical_decathlon/Task01_BrainTumour")
OUT_PATH = os.path.join("./output/")

if __name__ == "__main__":
    """
    Create a model, load the data, and train it.
    """
    """
    Step 1: Parse command line arguments
    """
    parser = argparse.ArgumentParser(
    description="2D U-Net model (Keras-Core) on BraTS Decathlon dataset.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d","--data_path", default=DATA_PATH,
                        help="The path to the Medical Decathlon directory")
    parser.add_argument("-o","--output_path", default=OUT_PATH,
                        help="the folder to save the model and checkpoints")
    parser.add_argument("--use_upsampling", type=bool,
                        help="use upsampling instead of transposed convolution",
                        action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="the batch size for training")
    parser.add_argument("-s","--split", type=float, default=0.80,
                        help="Train/testing split for the data")
    parser.add_argument("--crop_dim", type=int, default=128,
                        help="Size to crop images (square, in pixels). If -1, then no cropping.")
    parser.add_argument("-e","--epochs", type=int, default=30,
                        help="number of epochs to train")
    parser.add_argument("-lr","--learningrate", type=float,
                        default=1e-4,
                        help="learningrate")
    parser.add_argument("--weight_dice_loss", type=float,
                        default=0.85,
                        help="Weight for the Dice loss compared to crossentropy")
    parser.add_argument("--featuremaps", type=int,
                        default=16,
                        help="How many feature maps in the model.")
    parser.add_argument("--use_pconv", help="use partial convolution based padding",
                        action="store_true",
                        default=False)
    parser.add_argument("--channels_first", help="use channels first data format",
                        action="store_true", default=False)
    parser.add_argument("--print_model", help="print the model",
                        action="store_true",
                        default=True)
    parser.add_argument("--use_dropout",
                        default=True,
                        help="add spatial dropout layers 3/4",
                        action="store_true")
    parser.add_argument("--use_augmentation",
                        default=True,
                        help="use data augmentation on training images",
                        action="store_true")
    parser.add_argument("--output_pngs",
                        default="inference_examples",
                        help="the directory for the output prediction pngs")
    parser.add_argument("--input_filename",
                        help="Name of saved TensorFlow model directory")

    args = parser.parse_args()

    """
    Step 2: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the Medical Decathlon directory to a TensorFlow data loader ...")
    print("-" * 30)

    trainFiles, validateFiles, testFiles = get_decathlon_filelist(data_path=args.data_path, seed=args.seed, split=args.split)

    ds_train = DatasetGenerator(trainFiles, batch_size=args.batch_size, 
                                crop_dim=[args.crop_dim,args.crop_dim], 
                                augment=True, seed=args.seed)
    
    ds_validation = DatasetGenerator(validateFiles, batch_size=args.batch_size, 
                                     crop_dim=[args.crop_dim,args.crop_dim], 
                                     augment=False, seed=args.seed)
    
    ds_test = DatasetGenerator(testFiles, batch_size=args.batch_size, 
                               crop_dim=[args.crop_dim,args.crop_dim], 
                               augment=False, seed=args.seed)

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 3: Define the model
    """

    from utils.model import unet

    unet_model = unet(channels_first=args.channels_first,
                 fms=args.featuremaps,
                 output_path=args.output_path,
                 inference_filename=args.inference_filename,
                 learning_rate=args.learningrate,
                 weight_dice_loss=args.weight_dice_loss,
                 use_upsampling=args.use_upsampling,
                 use_dropout=args.use_dropout,
                 print_model=args.print_model)

    model = unet_model.create_model(
        ds_train.get_input_shape(), ds_train.get_output_shape())

    model_filename, model_callbacks = unet_model.get_callbacks()

    """
    Step 4: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    model.fit(ds_train,
              epochs=args.epochs,
              validation_data=ds_validation,
              verbose=1,
              callbacks=model_callbacks)

    """
    Step 5: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, ds_test)
