# Math-Written-to-Text-Conversion
## Motivation
This project aims to convert written math into on-screen text, preserving the relative location of math symbols on the page. This could be used to accurately transcribe mathwork into digital form with ease. This project hopes to accomplish this through three main stages.

## Project Pipeline
### Stage 1: Symbol Detection
Symbols that we want to transcribe must first be isolated for classification. Currently, this is done through the `findContours()` function in **OpenCV**. The image is first converted to a binary inverted image, so that contours detection is more accurate as per the documentation. Then, the resulting bounding boxes are used to fetch snippets of the original image, each containing a unique symbol. These snippets are fed into the second stage of this pipelined process.

### Stage 2: Symbol Classification
In order to know how to represent each symbol digitally, they must first be classified through a neural network. Specifically, each symbol snippet from the first stage is fed into a convolutional nueral network (CNN) that has already been trained to identify 82 different classes of symbols.

#### Training the CNN
The network was trained from scratch using **PyTorch** and its useful building blocks in the 'torch.nn' module.

1. The dataset used can be found [here](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols?resource=download). It contains thousands of images for each of 82 different classes. For simplicity, each class folder was reduced to 1000 images, 800 to be used for training and 200 to be used for testing, although these numbers are somewhat arbitrary and can be tweaked for future use. All images were loaded and converted to tensors using the `torchvision` library.
2. The CNN architecture used was the TinyVGG architecture for speed, since the dataset seemed simple enough to not warrant a larger network. For future use, other CNN models could be explored, such as VGG16 or VGG19.
3. The model was trained on 5 epochs and yielded a 92% accuracy on training data and a 90% accuracy on testing data. For better accuracy, the number of epochs and hidden units could be tweaked. The loss function and optimizer chosen could also be tweaked for improved accuracy and lower loss.

### Stage 3: Symbol Drawing
This stage uses the information from the previous two stages and draws text on a canvas that will be our output digital image. It takes the bounding box for each contour found in stage 1 and its classification in stage 2 to know what symbol to draw and where to draw it on the canvas. The **Pillow** library is used to accomplish this.

## Use Instructions
Currently, the project can only transcribe one image per run, although this can be easily fixed using the **os** library.

1. Put the image you want transcribed into the input folder with the name "target.extension".
2. Fetch the transcribed image from the output folder.

### Shortcomings and Future Improvements
Currently, the main problem with the project is its accuracy. While stage 2 and stage 3 seem to work as expected, stage 1 has some issues, specifically with contour detection. Contours detected are not always accurate, sometimes skipping over symbols or finding multiple contours in the same symbol. Contour detection doesn't work for symbols that require more than one stroke to draw and are not entirely connected, such as equality signs.

A solution for this is to do away with the simple CNN entirely, and get rid of the first stage. Instead, we will use a Region-based CNN, or R-CNN, that will scan an entire image and return bounding boxes along with classifications for each symbol. This will require more post-processing in stage 3 and will take longer to run, but will likely increase accuracy a lot. That is what I am working on currently.

Other solutions are welcome, as I am still hesitant on encorporating object detection + classification in this project if it isn't needed.