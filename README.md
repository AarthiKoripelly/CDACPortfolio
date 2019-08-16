# CDAC 2019 Portfolio
1. Laplacian Blob Detection
In computer vision, blob detection algorithms are common methods used to detect pixel brightness or color changes in a certain area of an image. Our approach here is to first review a common differential blob detector based on the Laplacian of Gaussian (LoG) function provided by the Python library skimage. Each image is sliced into 6 parts and the respective blob counts for added to an array.

    1) User Input Threshold Model:
	
		The user input threshold was made so users can insert their own focused or blurry count threshold. This allows for flexibility in the model. As of now, this model considers an image focused if three or more image slices had a blob count greater than 1000. This can be changed based on user preference.
	
        1. Principal Component Analysis 
		
            Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal translation to convert a set of observations, in this case six count values per image, of possibly correlated variables into a set of  linearly uncorrelated values known as principal components. This analysis was used to justify model one. 
			
    2) Neural Network Model
	
        The second model takes the array of the six blob values per image and trains a neural network to classify the images based on blob count.
    
2. Convolutional Neural Network Model

    The third model uses supervised learning that is capable of detecting blurry SEM images based on a relatively small data set. The total training data consists of roughly 900 images. However, since each image is over 3 billion pixels large, each one is randomly sliced into 14 300x300 images, giving us much dataset 14 times larger. There is another supplementary validation dataset consisting of 2,800 image slices (120 original SEM images). The input is a 300x300 numpy matrix, converted to double values through OpenCV. The output is a float value between 0 and 1, depicting the probability a given image is focused.
	
    The first layers consists of 124 filters of size 4 x 4 with a stride 1. Then, a max-pooling layer consists of a 4 x 4 kernel matrix. The max-pooling layer leads to faster convergence by selecting invariant features, which then improves generalization performance. The subsequent layer consists of a 68 filters of size 4 x 4, and then the next consists of a max-pooling layer of size 2 x 2. The fifth layer consists of 13 filters of size 2 x 2, and then the next consists of a max-pooling layer of size 2 x2. Each convolutional layer consists of a ReLU activation function. The final output layer is a fully connected layer with a sigmoid activation function in order to scale output values between 0 and 1. Adding a fully connected layer at the end allows for a way to learn non-linear combinations of high-level features outputted by the convolutional layer. 

3. DLHub Publishing

4. Ensembling Methods 
    1) Weighted Average
    2) Multi-feature Neural Network 



