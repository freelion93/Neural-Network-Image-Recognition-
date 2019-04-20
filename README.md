# Neural Network for MNIST object recognition

This neural network was built by me as one of the task for the optimization course. 
It uses [Fashion-MNIST dataset](http://yann.lecun.com/exdb/mnist/). Each image in the dataset is a 28x28 grayscale picture of the cloth, associated with a label from 10 classes. 
<p align="left"><b>Input image samples</b></p>

<p align="left"><img src="https://i.ibb.co/5rpsYCB/Screen-Shot-2019-04-21-at-01-07-40.png"></p>


<p align="left"><b>Labels</b></p>


|<span>|      |      |      |      |
|------|------|------|------|------|
|9     |2     |1     |1     |6     |
|1     |4     |6     |5     |7     |
|4     |5     |7     |3     |4     |
|1     |2     |4     |8     |0     |
|2     |5     |7     |9     |1     |

<p align="left"><b>Description</b></p>

|     <span>    |             |         |          |       |      |        |       |         |     |            |
|---------------|-------------|---------|----------|-------|------|--------|-------|---------|-----|------------|
|    *Lable*    |      0      |    1    |     2    |   3   |   4  |    5   |   6   |    7    |  8  |      9     |
| *Description* | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |

<p align="left"><b>Architecture</b></p>

|     Layer    | Neurons | Activation Function |
|:------------:|:-------:|:-------------------:|
|  Input Layer |   784   |          -          |
| Hidden Layer |   160   |       Sigmoid       |
| Output Layer |    10   |       Softmax       |
