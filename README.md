# BBM 409 Make-Up Assignment 

This is an assignment which requires implementing an Autoencoder using Neural Network.

To run the program succesfully model.py file must be in the same directory as the test.py and train.py files.
## Training

Use the below command to train the model.

```bash
python3 train.py -path_to_folder_containing_images_labels_of_dataset
```

## Testing

Important thing to know here is that all the pretrained .npy files should be in the same folders as the test.py file.

```bash
python3 test.py -path_to_txt_file_containing_names_of_npy_files