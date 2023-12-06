import os
import argparse
import logging
from glob import glob
import math
import random


def main():
    """Main function of the script."""

    SEED = 42

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+",
                        help="All the datasets to combine")
    parser.add_argument("--training_data_output", type=str,
                        help="path to training output data")
    parser.add_argument("--testing_data_output", type=str,
                        help="path to testing output data")
    parser.add_argument("--split_size", type=int,
                        help="Percentage to use as Testing data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.datasets)
    print("Training folder:", args.training_data_output)
    print("Testing folder:", args.testing_data_output)
    print("Split size:", args.split_size)

    train_test_split_factor = args.split_size / 100  # Alias
    datasets = args.datasets

    training_datapaths = []
    testing_datapaths = []

    for dataset in datasets:
        face_images = glob(dataset + "/*.jpg")
        print(f"Found {len(face_images)} images for {dataset}")

        # Concatenate the names for the face_name and the img_path. Don't put a / between, because the img_path already contains that
        # face_images = [(default_datastore, f'processed_faces/{face_name}{img_path}') for img_path in face_images # Make sure the paths are actual DataPaths

        # Use the same random seed as I use and defined in the earlier cells
        random.seed(SEED)
        random.shuffle(face_images)  # Shuffle the data so it's randomized

        # Testing images
        # Get a small percentage of testing images
        amount_of_test_images = math.ceil(
            len(face_images) * train_test_split_factor)

        face_test_images = face_images[:amount_of_test_images]
        face_training_images = face_images[amount_of_test_images:]

        # Add them all to the other ones
        testing_datapaths.extend(face_test_images)
        training_datapaths.extend(face_training_images)

        print(testing_datapaths[:5])

        # Write the data to the output
        for img in face_test_images:
            # Open the img, which is a string filepath, then save it to the args.testing_data_output directory
            with open(img, "rb") as f:
                with open(os.path.join(args.testing_data_output, os.path.basename(img)), "wb") as f2:
                    f2.write(f.read())

        for img in face_training_images:
            # Open the img, which is a string filepath, then save it to the args.testing_data_output directory
            with open(img, "rb") as f:
                with open(os.path.join(args.training_data_output, os.path.basename(img)), "wb") as f2:
                    f2.write(f.read())


if __name__ == "__main__":
    main()
