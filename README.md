# NFL Impact Detection

9th place Solution for the [NFL 1st and Future - Impact Detection Kaggle Competition](https://www.kaggle.com/c/nfl-impact-detection)

**Authors :** [Uijin Choi](https://github.com/Choiuijin1125), [Theo Viel](https://github.com/TheoViel), [Chris Deotte](https://github.com/cdeotte), [Reza Vaghefi](https://github.com/rvaghefi)

This repository contains the code for the 3D classification & post-processing parts of the pipeline.

## Context

> In this competition, you’ll develop a computer vision model that automatically detects helmet impacts that occur on the field. Kick off with a dataset of more than one thousand definitive head impacts from thousands of game images, labeled video from the sidelines and end zones, and player tracking data.

The aim of the competition is to detect helmet impacts in American football plays. For each play, we have access to two views : Endzone & Sideline, as well as the player tracking data. 


## Solution Summary 

> From https://www.kaggle.com/c/nfl-impact-detection/discussion/209012

### Overview
Our solution consists of a two-step pipeline (detection and classification) followed by some post-processing. We use 2D detection to find possible impact boxes, and then we use 3D classification to determine which boxes are impacts. 

#### Using 2D, Is This a Helmet Impact?

Our detection model finds the yellow box as a possible helmet impact.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1723677%2F941a0e534d355b76997947feae189af8%2Fquiz.png?generation=1609887772117223&alt=media" width="512">

From only 2D it is impossible to know whether these helmets are about to impact. It appears probable but maybe the silver helmet is about to pass behind the white helmet.

#### Using 3D, Is This a Helmet Impact?

Our classification model uses frames before and after this frame. With this information, our classification model correctly identifies that these two helmets do not impact but rather pass by each other. 

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1723677%2F681af4fc454c479b02a760dc82480396%2Fframes.png?generation=1609887810082337&alt=media" width="768">

### Impact Detection Model

> written by [Uijin Choi](https://github.com/Choiuijin1125)

We used DetectoRS(ResNeXt-101-32x4d) model for detecting impacts for [MMDetection](https://github.com/open-mmlab/mmdetection).  This model was built in two steps:

- Helmet Detection Model for warm-up: We trained a Helmet Detection Model for 12 epochs using the train helmet dataset which is in the image folder, this model shows `bbox_mAP_50 = 0.877` validation score.
- 2-Class Detection Model: We used the final weights of the Helmet Detection Model (as pretrained weights) to build a Helmet/Impact Detection model. We used  +/- 4 frames from impact as the positive class.

Using the Helmet Detection Model as pretrained weight makes the 2-Class Model converge much faster and detect impacts easier.

This approach showed good performance to detect impact. We set the confidence score up to 0.9 which showed 0.39 public LB score after a simple post processing (without any classifiers). Then, we set a lower confidence score to catch more true positives after we plug in the classifier. DetectoRS showed good performance to detect impact but it took a long time to test our ideas.

### Impact Classification Model
> written by [Theo Viel](https://github.com/TheoViel)

#### 2D Models

I kept struggling with improving my EfficientDet models as I had no experience with object detection, I figured out it would be better to go back to what I can do : classification models. The main idea was that a crop around a helmet has the same information regarding whether it is an impact or not as the whole image. 
Therefore, I extracted a 64x64 crop around all the boxes in the training data and started building models to predict whether a crop had an impact. To tackle imbalance, I used the [-4,  +4] extended label as a lot of people did. After a few tweaking, I had a bunch of models with a 0.9+ AUC (5 folds stratified by gameplay) : Resnet-18 & 34, EfficientNet b0 -> b3. 

Tricks used for 2D models include :
Limiting the number of boxes sampled per player at each epoch, in order to have more control on convergence
Linear learning rate scheduling with warmup
Removing the stride of the first layer : 64x64 is a small image size and it’s better not to decrease the network resolution too fast
Stochastic Weighted Averaging
Classical augmentations 

Then, I used [@tjungblut’s detector](https://www.kaggle.com/c/nfl-impact-detection/discussion/200995)  to benchmark how bad my models were on the public leaderboard. Turns out that after some post-processing, a resnet-18 + efficientnet-b1 + efficientnet-b3 blend achieved 0.33+, which at that time was in the gold zone.

#### Merging

Shortly after, I merged with the rest of the team with the goal of plugging my classification models on top of a strong detection model to leverage their potential. There were about two weeks left before the end of the competition, so we first focused on plugging @Uijin’s detection models with my classifiers. For a while, we couldn’t beat the detector LB’s of 0.39+, but after setting up a good CV scheme and improving the detection model training strategy, we reached LB 0.41+. 

#### 3D Models 

Around this time, I upgraded my 2D classifiers to be 3D. Models will now take as input frames [-8, -6, -4, -2, 0, 2, 4, 6, 8] instead of just frame 0. The pipeline was easy to adapt, the data loader was slightly modified, architectures were changed, and augmentations were removed for I was too lazy to dig into that.

The first batch of 3D model is based on [3D Resnets](https://github.com/kenshohara/3D-ResNets-PyTorch) : 
- resnet-18 & resnet-34, using as target the same as the one of the middle frame (that was extended)
- resnet-18, using an auxiliary classifier to predict the impact type
- resnet-18, with a target extended to frames [-6, +6]

The only additional trick I used is getting rid of all the strides on the temporal axis. Models are made for videos longer than 9 frames so once again I adapt the networks to my small input size.

They helped CV, didn’t really help public LB. In fact our jump to 0.49 at the time came from retraining our detection model on the whole dataset & tweaking some post-processing parameters. 
They did however help private LB, but this was after the leak so we didn’t know. 

#### More 3D Models

After recovering from NYE, I did some state of the art browsing and implemented 3 new 3D models that all outperformed the previous ones : 
[i3d](https://github.com/piergiaj/pytorch-i3d), [Slowfast](https://github.com/open-mmlab/mmaction2) and [Slowonly with Omnisource pretrained weights](https://github.com/open-mmlab/mmaction2).

This was done on the 2nd and 3rd of January, so we had 10 submissions to try these out because of my procrastinating. My first submissions using them gave a small LB boost and we reached 0.50+. 

Fortunately the rest of the team worked hard on the detection models to compute results of our detection models on all our folds. This allowed them to find a powerful set of hyperparameters which worked in the end !

### Validation - "Trust Your CV"
> written by [Chris Deotte](https://github.com/cdeotte)

Building a reliable validation was very important because it was easy to overfit the public LB or a hold out validation set of only a few videos. Our final model was configured using a full 120 video 5 fold validation. By reviewing our OOF predictions, we were able to tell what we needed to do to improve our model. One tool we used was to watch the ground truths and predictions together in all 3 views. The code is available [here](https://www.kaggle.com/c/nfl-impact-detection/discussion/208782)

[<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1723677%2Fe46b13bc1feb0e773fb476e689cf5403%2Fpreview.png?generation=1609809137273907&alt=media" width="512">](http://playagricola.com/Kaggle/labeled_57586_001934-pair_1.mp4)

This gave us insight into choosing hyperparameters for our detection and classification models and it gave us insight into creating post process.

### Post-processing
> written by [Reza Vaghefi](https://github.com/rvaghefi)

#### Thresholding

We had to find good thresholds for both our detection and classification models. We observed that a single split is not enough and F1 score variation between folds is significant. Therefore, we used the CV calculated on the entire train set (5 folds) to optimize our thresholds. The following tricks improved our CV and LB:
Different thresholds for detection and classification model: Originally we had the same threshold for detection and classification models (~0.5). As the classification model became better, we needed to lower detection thresholds (~0.35) to include more helmet detections and then use the classification model to classify the impacts.
Different thresholds for Endzone and Sideline views: Endzone and Sideline videos are different in terms of image content, box area, box size, etc. We realized that using different thresholds can improve both CV and LB. We tried different combinations and best CV was achieved by using around 0.05 higher threshold in Sideline than Endzone
Different thresholds over time: Both detection and classification models have no information about time-elapsed. From the training set, we know that the chance of impact decreases as frame number increases. We tried different combinations (fixed, piecewise, and linear) and ended up using a piecewise method where we used different thresholds for frame >= 150 and frame < 150.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1723677%2F73d38ec1aa72b0375c3106e4ba1c4c0b%2Ftime2.png?generation=1609892194122820&alt=media" width="512">


#### Box Expansion

Increasing the size of the bounding boxes helped our models, for some reason. In our last submission, we made the bounding boxes 1.22x bigger.

#### Adjacency Post-Processing

The goal of this PP was to cluster detections (through multiple frames) which belong to the same player and remove FP. We used an IOU threshold and frame-difference threshold to optimize our algorithm. We also tried temporal NMS and soft-NMS but our custom algorithm performed better. 

#### View Post-Processing

Impacts that were detected with a confidence lower than `T` were removed if no impact was found in the other view. That’s the best we came up with regarding merging information from both views.

### Final results

We selected two submissions : Best public LB and best CV.
- CV 0.4885 - **Public 0.5408 (10th)** - Private 0.4873 (13th)
- **CV 0.5125** - Public 0.4931 (13th) - **Private 0.5153 (9th)**

Our competition inference notebook is available [here](https://www.kaggle.com/cdeotte/nfl-2d-detect-3d-classify-0-515).

## Data

- Competition data is available [on the Kaggle competition page](https://www.kaggle.com/c/nfl-impact-detection/data)
- Model weights are available [on this Kaggle dataset](https://www.kaggle.com/theoviel/nfl-dataset-3)

## Repository structure

The reprository organization and its main components is detailed bellow :

- `mmaction2/` : MMaction2 repository, to avoid fully installing the package which can be a bit tricky.

- `notebooks/` : Notebooks to perform data preparation, training & inference
  - `Data preparation.ipynb` : Extracts images from videos
  - `Data preparation Cls 3D.ipynbb` : Extracts 3D helmet crops from images
  - `Inference Cls.ipynb` : Performs inference
  - `Training Cls 3D.ipynb` : Trains a 3D classification model
  
- `output/` : Outputs of the training
  - `22_12/` : Results of the detection model
  - `folds.csv` : Cross-validation folds
  - `df_preds.csv` : 3D classifier predictions on the best detection model
  
- `src/` : Source code
  - `data/` : Datasets and data preparation functions
  - `inference` : Functions for inference
  - `model_zoo` : 3D classification Models
  - `post_processing` : Adjacency, view and expansion post-processing 
  - `training` : Training functions
  - `utils` : Logger, torch utils and metric
  - `configs.py` : Model configs of the final blend
  - `params.py` : Global parameters


## Training a model

Training a model is done in the notebooks. Paths have to be updated in `src/params.py`. 

- First download the competition data from [Kaggle](https://www.kaggle.com/c/nfl-impact-detection/data).
  - Specify `DATA_PATH` and `TRAIN_VID_PATH` accordingly to where you store the data
  
- Run the `Data preparation.ipynb` notebook
  - This extracts the frames from the videos and computes a training dataframe
  - Specify `IMG_PATH_F`  accordingly to where you want to save the images
  
- Run the `Data preparation.ipynb` notebook
  - This extracts 3D crops around the training helmets
  - Specify `CROP_PATH_3D`  accordingly to where you want to save the 3D crops
  
- Run the `Training Cls 3D.ipynb` notebook
  - Specify the `Config` you want to use. The ones used in the final ensembles are in `src/configs.py`
  - Specify `LOG_PATH_CLS_3D` accordingly to where you want  to log results

Evaluation of the model is done in the `Inference Cls.ipynb` notebook.
You can re-use the pre-computed predictions `preds.csv`, or compute new ones by specifying `CP_FOLDER` and `configs` in the `Classifier 3D inference` section.
