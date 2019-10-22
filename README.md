###  Transfer-Learning-in-General-Lensless-Imaging-through-Scattering-Media

##### Requirements:

- ##### python2.7

- ##### tensorflow 1.13.1

##### Dataset Download:

###### https://cloud.tsinghua.edu.cn/d/ada3bbbb0c91472dbb90/

##### Getting Started:

###### For training: 

```
cd cifar/mnist
cd training_code
python Train.py 
```

###### For fine-tuning: 

```
cd cifar/mnist
cd transferlearning_code
python trans_mnist.py/trans_cifar.py
```

Please note that you should change the model path when you run trans_mnist.py/trans_cifar.py.

