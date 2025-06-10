# Automatic-Classification-of-Nutritional-Deficiencies-in-Coffee-Leaves-Images-Using-Transfer-Learning


The focus of this study is on the **automatic detection of nutritional deficiencies** such as **Potassium, Boron, Calcium, and Iron** in coffee plants using **transfer learning techniques** — MobileNetV2, InceptionV3, VGG19, EfficientNetV2, and ResNet50 — and **data augmentation strategies**, using image data from the **CoLeaf Dataset V2.0**.  
Our aim is to **improve early diagnosis** of deficiencies, which can significantly affect **crop quality and yield**.

All of the datasets are available within this repository inside `./src/Datasets` directory and all python scripts within the `./src` directory are reusing this local dataset.

## Problem Statement

Coffee plants require vital nutrients like Boron, Calcium, Iron, and Potassium for healthy growth. Deficiencies manifest in the leaves and are challenging to identify manually. We propose an automated system using deep CNN models and image augmentation to classify the deficiencies effectively.

---

## Methodology Overview

The following models were explored:
- **MobileNetV2**
- **InceptionV3**
- **VGG19**
- **EfficientNetV2**
- **ResNet50**

Dataset: **CoLeaf Dataset V2.0**, 1006 annotated leaf images.

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



## Flowchart of Proposed Work

The image below illustrates the pipeline used in this study — from data preprocessing to classification using transfer learning:

![Flowchart of Proposed Work](https://raw.githubusercontent.com/sarath1102/Automatic-Classification-of-Nutritional-Deficiencies-in-Coffee-Leaves-Images-Using-Transfer-Learning/main/flowchart.png)


