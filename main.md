# SENIOR PROJECT : Neural Artistic Style Transfer

* [Notes](./notes.md)
* [Vocabulary](./vocab.md)
* [Experiment](./experiment.md)

## What is a Neural Artistic Style Transfer?

Neural artistic style transfer is a method of recreating images in the style of other images, by use of a neural network. The images are not merged together. Essentially, you alter a picture as though someone else created it - as if an image's type is casted onto a different image. 

I should spin off of the existing projects. I can write a better algorithm or script for it - but I doubt that I am smart enough. Or I can make it do something different and specialized - like faces only. I can turn the face into a different landscape.

### Relevant Sources
* [CODAME](http://codame.com/events/workshop-visual-strategies-for-neural-artistic-style-transfer)
* [Gary's Github](https://github.com/skinjester/visual-strategies/wiki)
* [Gary's Projects](https://www.deepdreamvisionquest.com/)
* [Grafitti Project](https://medium.com/s/story/digital-processes-inspiring-analog-paintings-a358eb7801a0)
* ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)
* [Intro to NN & Deep Learning](https://skymind.ai/wiki/neural-network)
* [Example Keras Notebook](https://github.com/kenophobio/keras-example-notebook)
* [Build Your own NN](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
* 3Blue1Brown Youtube channel S3

### Contact
* talked to Professor Ventura
* Clint staley deep learning class?
* Contact Gary Boodhoo

### Materials & Need to Know
* Google Colaboratory (run 1000 times - remote graphics server)
* take a look at keras notebooks

### QUESTIONS

* Regarding pooling and parameter reduction - I understand that we want to avoid overfitting (learning unecessary features) but when you reduce/compress spatial size, do you lose important information? Do you distort data?
    * larger weights = larger importance -> we keep the more important data and neglect the lesser data to avoid unecessary computations?


* **important point from http://cs231n.github.io/convolutional-networks/#layers**
    * _""Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch."" _
