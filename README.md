
## Install Anaconda Environment

```
conda create --name adCVPR18 --file environment.yml -c defaults -c pytorch -c conda-forge -y
conda activate adCVPR18
```

## Feature Extractor Weights

### C3D
https://drive.google.com/file/d/1Ev5ZIYuOIWHHO6SujDeR1eugNnvO2hWv/view?usp=sharing

### R3D-101
https://drive.google.com/file/d/1p80RJsghFIKBSLKgtRG94LE38OGY5h4y/view?usp=share_link

### R3D-152
https://drive.google.com/file/d/1irIdC_v7wa-sBpTiBlsMlS7BYNdj4Gr7/view?usp=share_link

## Precomputed Features
Can be downloaded from:

### C3D features
https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50?usp=sharing

### ResNet-101 features (by @Daniele Mascali)
https://drive.google.com/file/d/1kQAvOhtL-sGadblfd3NmDirXq8vYQPvf/view?usp=sharing

### ResNet-152 features (by @Henryy-rs)
https://drive.google.com/file/d/17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK/view

## Pre-Trained Anomaly Detector
Check out <a href="exps/">exps/</a> for for trained models on the pre-computed features

## Features Extraction
Download the dataset from: https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
Arguments:
* dataset_path - path to the directory containing videos to extract features for (the dataset is available for download above)
* model_type - which type of model to use for feature extraction (necessary in order to choose the correct pre-processing)
* pretrained_3d - path to the 3D model to use for feature extraction

```python feature_extractor.py --dataset_path "path-to-dataset" --model_type "fe-model-eg-c3d" --pretrained_3d "path-to-pretrained-fe"```

## Training
Arguments:
* features_path - path to the directory containing the extracted features (pre-computed features are available for download above, or supply your own features extracted from the previous stage)
* annotation_path - path to the annotations file (Available in this repository as `Train_annotations.txt`)

```python TrainingAnomalyDetector_public.py --features_path "path-to-dataset" --annotation_path "path-to-train-annos"```

## Generate ROC Curve
Arguments:
* features_path - path to the directory containing the extracted features (pre-computed features are available for download above, or supply your own features extracted from the previous stage)
* annotation_path - path to the annotations file (Available in this repository as `Test_annotations.txt`)
* model_path - path to the trained anomaly detection model

```python generate_ROC.py --features_path "path-to-dataset" --annotation_path "path-to-annos" --model_path "path-to-model"```

I achieve this following performance on the test-set. I'm aware that the current C3D model achieves AUC of 0.69 which is worse than the original paper. This can be caused by different weights of the C3D model or usage of a different feature extractor.

| C3D (<a href="exps\c3d\models\epoch_80000.pt">Link</a>) | R3D101 (<a href="exps\resnet_r3d101_KM_200ep\models\epoch_10.pt">Link</a>) | R3D152 (<a href="exps\resnet_r3d152_KM_200ep\models\epoch_10.pt">Link</a>) |
| :-----------------------------------------------------: | :------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
|      <img src=graphs/roc_auc_c3d.png width="300"/>      |               <img src=graphs/roc_auc_r101.png width="300"/>               |               <img src=graphs/roc_auc_r152.png width="300"/>               |


## Demo

### Off-line (with video loader)
Arguments:
* feature_extractor - path to the 3D model to use for feature extraction
* feature_method - which type of model to use for feature extraction (necessary in order to choose the correct pre-processing)
* ad_model - path to the trained anomaly detection model
* n_segments - the number of segments to chunk the video to (the original paper uses 32 segments)

```python video_demo.py --feature_extractor "path-to-pretrained-fe" --feature_method "fe-method" --ad_model "path-to-pretrained-ad-model" --n_segments "number-of-segments"```

The GUI lets you load a video and run the Anomaly Detection code (including feature extraction) and output a video with a graph of the Anomaly Detection prediction below.

**Note**: The feature extractor and the anomaly detection model must match. Make sure you are using the anomaly detector that was training with the corresponding features.

### On-line (via webcam)
Arguments:
* feature_extractor - path to the 3D model to use for feature extraction
* feature_method - which type of model to use for feature extraction (necessary in order to choose the correct pre-processing)
* ad_model - path to the trained anomaly detection model
* clip_length - the length of each video clip (in frames)

```python AD_live_prediction.py --feature_extractor "path-to-pretrained-fe" --feature_method "fe-method" --ad_model "path-to-pretrained-ad-model" --clip_length "number-of-frames"```

The GUI lets you load a video and run the Anomaly Detection code (including feature extraction) and output a video with a graph of the Anomaly Detection prediction below.

**Note**: The feature extractor and the anomaly detection model must match. Make sure you are using the anomaly detector that was training with the corresponding features.

*Contrbuted by Peter Overbury of Sussex Universty IISP Group

## Cite
```
@misc{anomaly18cvpr-pytorch,
  author       = "Eitan Kosman",
  title        = "Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos",
  howpublished = "\url{https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch}",
  note         = "Accessed: 20xx-xx-xx"
}
```

## FAQ
1.
```
Q: video_demo doesn't show videos
A: Downlaod and install LAVFilters: http://forum.doom9.org/showthread.php?t=156191
```

2.
```
Q: What is the meaning of the second column of Train_Annotations.txt?
A: Length of the video in frames. Note that it has not effect on training. It exists because these are the original annotations supplied by the authors.
```
