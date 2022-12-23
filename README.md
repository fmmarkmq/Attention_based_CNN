# Attention_based_CNN
In the previous project, we were trying to implement a CNN-based approach to predict panel time series data. CNN-based approaches work very well in image processing since the adjacent pixels should have the greatest causal impact on their surrounding pixels, which can be captured by a convolutional kernel.

However, in the time-series panel data, it’s not necessarily true that the adjacent observations have the greatest causal impact.

Also, we read some research indicating a smaller CNN kernel would yield better performance. For example, in the classic MNIST dataset, a 3x3 kernel works very well. So we investigated further the reasoning behind this. We hypothesized the 3x3 kernel captures surrounding pixels with relatively the largest covariance with respect to the center pixel. So we verified our thoughts by plotting the covariance relationship (show him the picture)

This inspired us to come up with an algorithm that has a small yet flexible kernel. For each convolution, the algorithm should look for other pixels with maximum covariance across the entire dataset as the kernel. This approach is very similar to the concept of the attention mechanism. 

We hypothesize that this approach can maximize the effective receptive field while minimizing kernel size and thus minimizing the number of parameters, which helps the model converge faster when given limited observations.

We also hypothesize that this approach can be generalized to other types of input data beyond panel time series data.
