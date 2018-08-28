
# YoloV3 in Pytorch and Jupyter Notebook
<p align="center">
  <img width="630" height="350" src="x_wing.gif">
</p>

This repository aims to create a YoloV3 detector in **Pytorch** and **Jupyter Notebook**. I'm trying to take a more "oop" approach compared to other existing implementations which constructs the architecture iteratively by reading the config file at [Pjreddie's repo](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg). The notebook is intended for study and practice purpose, many ideas and code snippets are taken from various papers and blogs. I will try to comment as much as possible. You should be able to just *Shift-Enter* to the end of the notebook and see the results.

## Requirements

 - Python 3.6.4
 - Pytorch 0.4.0
 - Jupyter Notebook 5.4.0
 - OpenCV 3.4.0
 - Cuda Support

## Instructions
    $ git clone https://github.com/ydixon/yolo_v3
    $ cd yolo_v3
##### Download Yolo v3 weights
	$ wget https://pjreddie.com/media/files/yolov3.weights
##### Download Darknet53 weights
	$ wget https://pjreddie.com/media/files/darknet53.conv.74
##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh

## How to use
These notebooks are intended to be self-sustained as possible as they could be, so you can just step through each cell and see the results. However, for the later notebooks, they will import classes that were built before. It's recommended to go through the notebooks in order.
### yolo_detect.ipynb
This notebook takes you through the steps of building the darknet53 backbone network, yolo detection layer and all the way up to objection detection from scratch.
<pre>
&#8226; Conv-bn-Relu Blocks				&#8226; Residual Blocks
&#8226; Darknet53					&#8226; Upsample Blocks
&#8226; Yolo Detection Layer				&#8226; Letterbox Transforms
&#8226; Weight Loading				&#8226; Bounding Box Drawing
&#8226; IOU - Jaccard Overlap				&#8226; Non-max suppression (NMS)
</pre>
### COCODataset.ipynb
Shows how to parse the COCO dataset that follows the format that was used in the original darknet implementation .
<pre>
&#8226; Generate labels			&#8226; Image loading
&#8226; Convert labels and image to Tensors	&#8226; Box coordinates transforms
&#8226; Build Pytorch dataset			&#8226; Draw
</pre>
### yolo_train.ipynb
Building up on previous notebooks, this notebook implements the back-propagation and training process of the network. The most important part is figuring out how to convert labels to target masks and tensors that could be trained against. This notebook also modifies `YoloNet` a little bit to accommodate the changes.
<pre>
&#8226; Multi-box IOU 			&#8226; YoloLoss
&#8226; Build truth tensor			&#8226; Generate masks
&#8226; Loss function				&#8226; Differential learning rates
&#8226; Intermediate checkpoints		&#8226; Auto-resume training
</pre>
**Need more work. Would add data-augmentation and implement better metrics to evaluate the model. The model should start see reasonable results around 10 epochs**
## TODO
  1. mAP (mean average precision)
  2. Data augmentation (blur, random flip)
  3. Implement backhook for YoloNet branching
  4. Color palette for bounding boxes
  5. Imaging saving
  6. Feed Video to detector
  7. Fix possible CUDA memory leaks
  8. Fix class and variable names


## References

1. [YOLOv3_: An Incremental Improvement. Joseph Redmon, Ali Farhadi ](https://pjreddie.com/media/files/papers/YOLOv3.pdf) 
2. [Darknet Github Repo](https://github.com/pjreddie/darknet)
3. [Fastai](http://www.fast.ai/)
4. [Pytorch Implementation of Yolo V3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
5. [Deep Pyramidal Residual Networks](https://arxiv.org/abs/1610.02915)
6. [eriklindernoren's Repo](https://github.com/eriklindernoren/PyTorch-YOLOv3)
