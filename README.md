
Definitions, Acronyms, or Abbreviations 
CUDA - CUDA is a parallel computing platform and application programming interface (API) model created by NVIDIA 
CuDNN - NVIDIA CUDA Deep Neural Network library is a GPU-accelerated library of primitives for deep neural networks 
Theano - Theano is a numerical computation library for Python 
Keras - is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano 
2. Important files 
Main.py - this is the main source file which contains all interpreted code. 
 
Hardware Configuration 
Processor: 	Core i5-4210H CPU @ 2.90GHz, 2901 Mhz, 2 Core(s), 4 Logical 
GPU:   	Nvidia GeForce GTX 860M RAM:  	8 GB DDR3 

System Parameters 
Operation system: 	Windows 10; x64 
Python: 	  	2.7 version 
Theano: 	 	0.8 version 
CuDNN: 	 	5.4 version 
Cuda:  	 	7.5 version 
Theano parameters: floatX=float32, device=gpu, nvcc.fastmath=True, lib.cnmem=0.75, optimizer=cudnn 
cnmem - CUDA optimal memory allocation 
Nvcc.fastmath - optimizing Nvidia C compiler to compute math faster 
 
Operation Procedure: 
1.	Go to the directory with main.py file 
2.	Run main.py file using python 2.7 interpeter 
Note: Theano run parameters can be changed it Windows Env Variables. Experienced users can change # of epochs, layers sequence and tune other parameter in the source code. 

DataSet:
The CIFAR-10 dataset is an established computer vision dataset collected by Alex Krizhevsky. It consists of 60,000 32x32 colorimages in 10 classes, with 6000 images per class. There are 50,000 training images and 10000 test images. 

Experiment:
I experimented on several popular activation functions and tweaked on layer organization. From the experiments results comparison, we found when selecting activation function for the convolution neural network, hyperbolic tangent is slightly better than sigmoid function. However, ReLu is the best. It helps training the neural network faster and has good accuracy.
In terms of accuracy, the smaller size to convolve and the more number of filters, the more accurate results are. This is simply because more detail information can be captured. Meanwhile, fully connected layers also helps to consolidate and utilize fully the current information. Therefore, it is not surprising in the above example, model 5 and 6, the models has one more fully connected layer, achieve accuracy improvement. The tradeoff is it the training time is increasing as well with the number of filters and the fully connected layer.
In addition, normalization is proved to be efficient in improving accuracy but the time increases as well. Model 6 is implemented with a normalization layer after first round of convolution and max pooling. The accuracy is improved by 5% but the time is almost doubled. Due to the time 
and accuracy tradeoff, a lightweight convolutional neural network deployed with ReLu activation function can achieve decent accuracy within a reasonable good amount of time. Many optimizations such as stochastic gradient decent with cross-entropy loss function, and  normalization help boost up the performance.
