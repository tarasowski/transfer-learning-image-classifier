# Transfer Learning Image Classifier

This project uses the transfer learning approach to classify images.
According to Andrew Ng the transfer learning will be the next driver of ML
commercial success.

## Pre-requisites
* Python v3.7
* Torch v1.4.0
* Trochvision v0.5.0
* Images (Train/Valid/Test)- [download here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)

## Getting Started
This project has two main files. The file for the training part and the prediction (inference) part. The training part is generic and can be used with any pre-trained model from the [Torchvision module](https://pytorch.org/docs/stable/torchvision/models.html). As a starting point, you can use `densenet121`. 
First, you need to download the images for the training, validation and testing phase (see link above). Unzip the images and place them into the `./input` directory. 

Once the images are unziped, you can use following arguments to train the model: 

* Basic usage: `./train data_directory`
* Options:
  * Set directory to save checkpoints: `./train data_directory --save_dir save_directory`
  * Choose architecture: `./train data_directory --arch densenet121`
  * Set hyperparameters: `./train data_directory --learning_rate 0.003 --hidden_units 512 --epochs 10`
  * Use GPU for training: `./train data_directory --gpu`
* Example: `./train ./input/flowers/ --arch densenet121`  

For the prediction part please use following arguments: 

* Basic usage: `./predict path/to/image checkpoint`
* Options:
  * Return top `K` most likely classes: `./predict path/to/image --top_k 5`
  * Use a mapping of categories to real names: `./predict path/to/image
checkpoint --category_names cat_to_name.json`
  * Use GPU for inference: `./predict path/to/image checkpoint --gpu`
* Example: `./predict ./input/flowers/test/102/image_08012.jpg checkpoint --top_k 3 --category_names cat_to_name.json`

## Support
Patches are encouraged and may be submitted by forking this project and submitting a pull request through GitHub.

## Credits
The project was developed during the ML program of [Udacity.com](https://www.udacity.com/)

## Licence
Released under the [MIT License](./License.md)
