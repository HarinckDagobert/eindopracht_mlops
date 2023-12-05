import os


def rename_files(folder_path, new_prefix):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file and rename it
    for file in files:
        old_path = os.path.join(folder_path, file)

        # Extract the file extension
        _, extension = os.path.splitext(file)

        # Create the new file name based on the specified pattern
        new_name = f"{new_prefix}_{file.split('_')[-1]}"

        # Construct the new file path
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} to {new_name}")


# Specify the folder path and new prefix
# Replace with the actual folder path
folder_path = "C:/school/jaar 3/semester 1/MLOps/eindopracht_mlops/data/train/surprise/"
new_prefix = "surprise"

# Call the function to rename files
rename_files(folder_path, new_prefix)
