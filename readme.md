# Siamese Network For Style Classification
This project use Siamese Network for the design style classification job.
Now it can distinguish "modern" and others.

# Data
Data is not available because it is provided by Tezign Tongji Design AI Lab.(sheji.ai & design-net.org)

# Model
This work is inspired by *Siamese neural networks for one-shot image recognition*[1].
Model in this work is different from model in this paper.  
This work uses pre-trained ResNet34[2] as the basic, get the features from the last convolutional layer and add a fully connected layer. There are 2 this structure in siamese network with shared weight, followed by a fully connected layer(get 2 inputs at a time) and sigmoid function.

# Result
The result looks good on training and test set, but verification on the large-scale design dataset is not available due to the lack of the data.  
Here are some results for "modern". In each line there are two pictures, the first one is in modern style, and the second one can be the same or others. Text above the images shows the label annotated by human(1 for same style and 0 for different style), results predicted by this model and the label of the second image.  
![image](https://raw.githubusercontent.com/zhuojg/siamese_network_for_style/master/result/result1.png)  
![image](https://raw.githubusercontent.com/zhuojg/siamese_network_for_style/master/result/result2.png)

# Reference
[1] Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." ICML deep learning workshop. Vol. 2. 2015.  
[2] The pre-trained model is provided by pytorch, https://download.pytorch.org/models/resnet34-333f7ec4.pth
