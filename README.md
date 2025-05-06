# Automatic-Classification-of-Nutritional-Deficiencies-in-Coffee-Leaves-Images-Using-Transfer-Learning

The focus of this study is on the automatic detection of nutritional deficiencies such as Potassium, Boron, Calcium, and Iron in coffee plants using transfer learning techniques — MobileNetV2, InceptionV3, VGG19, EfficientNetV2, and ResNet50 — and data augmentation strategies. 



All of the datasets are available in " https://data.mendeley.com/datasets/yy2k5y8mxg/1 " within this repository inside `./src/Datasets` directory and all python scripts within the `./src` directory are reusing this local dataset.




https://data.mendeley.com/datasets/yy2k5y8mxg/1
## Instructions for Executing the Code


1 - Fork this repo.

2 - Go to the `src` directory

3 - Each file executes a different experiment

    Execute 'Experiment-1' file to execute the Original Datasets.

    Execute 'Experiment-2' file to execute the Original Datasets with 
    Data Augmentation.

    Execute 'Experiment-3' file to execute the Oversampling Original Datasets.

    Execute 'Experiment-4' file to execute the Undersampling Original Datasets.

    Execute 'Experiment-5' file to execute the Oversampling Data Augmented Datasets.

4 - The additional python files are for executing Data Augmentation, Oversampling techniques.