---
layout: default
title: "07 Detection and Segmentation"
permalink: /vision/CS231N/07
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 7. Object Detection, Image Segmentation, Visualizing
## Vision Transformer

#### Architecture
- image patching and tokenization 
	- images are divided into 3x3 patches
	- each patch is projected into a token 
- positional embeddings 
	- patching loses spatial information . . . 
	- positional embedding for retaining location data 
- transformer layers 
	- multi-head self-attn, layer normalization, MLP 
	- do not require masking becuase the entire image is processed at once ! (language와의 차이점)
- output 
	- image classification : special class token is added (predict class probabilites)

#### optimization and tweaks
layer normalization
- standard 
- placed before self-attn 
- preserves the ability to learn an identity function (crucial for DL)

RMSNorm
- alternative norm
- simpler than standard layer norm
- improve training stability 

Gated MLPs (SwiGLU)
- uses a gated non-linearity by introducing a 3rd weight matrix in the MLP
- learn higher-dimensional non-linearities without significantly increasing the number of params

MoE 
- instead of single MLP - multiple expert MLPs are used 



## core computer vision tasks 
### semantic segmentation 

challenge of pixel-level classification
- classifying pixel-wise is difficult (lack of context)
- CNN can be trained to consider surrounding pixels for classification (patch-wise)

#### fully Convolutional Networks (FCNs)
basic 
- input : image / output : entire segmentation map (assign a label to each pixel)
- maintain spatial resolution, avoiding fully connected layers that would lose this information 
- problem : large images → massive numbers of params 

encoder-decoder architecture
- FCNs use encoder-decoder structure (to manage computational cost)
- downsampling 
	- progressively reduces spatial resolution while increasing the number of channels 
	- pooling, strided convolution 
- upsampling 
	- reconstructs the image resolution to produce the final segmentation map
	- unpooling, nearest neigbor, bed of nails (1 pixel, others zero), learned upsampling (learnable params) 

U-Net Architecture
- popular FCN variants with a U-shaped architecture
- skip connection (copy feature maps from the encoder to the corresponding decoder layers)


### object detection 
#### single object
- goal : predict both the class label & bounding box coordinates ($x, y$, width, height) for each object 
- multitask loss can be defined 
- softmax loss (classification) + L2 loss (bounding box regression)

#### multiple objects
- scalability : generating predictions for every possible bounding box is computationally infeasible 
- each image needs a difference number of output ! 

#### region proposal methods 
RPNs (Region Proposal Networks)
- identify 'region of interest (ROI)' that are likely to contain objects

RCNN (region-based CNN)
- generates region proposals 
- CNN on each proposed region to classify it and refine its bounding box
- problem : very slow  

Other RCNN
- Fast RCNN : entire image once and extract features for each region proposal from the resulting feature map 
- Faster RCNN : RPN directly into the CNN (single, end-to-end trainable system)
- Mask RCNN : predicting pixel-wise mask for each detected object. Faster RCNN + parallel branch 

Single-Stage Detectors
- perform classification and bounding box regression in a single pass (faster)
- YOLO (You only look once)![[Pasted image 20260329105643.png]]

Transformer-based Detectors (DETR)
- use transformers for object detection 
- image is processed by CNN to create feature maps 
  → converted into tokens with positional encodings 
  → fed into a transformer encoder 
  → transformer decoder takes the encoder output and a set of learned object queries as input 
  → each query learns to predict a specific object's class and bounding box 
- eliminates the need for hand-designed components (e.g. anchor box, non-maximal suppression)

#### visualizing filters 
- early CNN layers - filters often learn basic patterns (edge, orientation, simple shapes)
- visualized directly if the # of channels is small
- as layers get deeper, filters learn more complex and holistic patterns 

#### saliency maps 
- highlight the pixels in an image that are most important for a network's prediction 
- how? : by calculating the gradient of the loss / class score with respect to the pixel values 
- pixels with larger gradients = changing their values would significantly affect the prediction 

#### class activation maps (CAM, Grad-CAM)
CAM (Class Activation Mapping)
- use the weights learned on the feature maps to create a heatmap
- show which regions of the image contributed most to a specific class prediction
- limitation : only applicable to the last Convolutional layer 

Grad-CAM (Gradient-weighted CAM)
- more general apporach 
- visualization of activations deeper within the network 
- produces heatmaps that highlight important regions for a given class

#### transformer visualization 
- Vision Transformers (VITs) = provide attentio maps 
- show how difference parts of the input relate to each other and to the output 
- understanding the model's focus 






