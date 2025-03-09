// questions-database.js
// This file stores all quiz questions in a structured format

const questionsDatabase = {
  "singleChoice": [
    {
      "question": "SC Q1: Which function is typically used as the final layer in a multi-class classification neural network?",
      "options": ["ReLU", "Sigmoid", "Softmax", "Tanh"],
      "answer": "Softmax"
    },
    {
      "question": "SC Q2: Which activation function is most known for mitigating the vanishing gradient problem?",
      "options": ["Sigmoid", "Tanh", "ReLU", "Softmax"],
      "answer": "ReLU"
    },
    {
      "question": "SC Q3: Which weight initialization method is best suited for networks using ReLU activations?",
      "options": ["Xavier initialization", "He initialization", "Random initialization", "Zero initialization"],
      "answer": "He initialization"
    },
    {
      "question": "SC Q4: Which technique prevents overfitting by randomly disabling neurons during training?",
      "options": ["Batch normalization", "Data augmentation", "Dropout", "Early stopping"],
      "answer": "Dropout"
    },
    {
      "question": "SC Q5: Which technique normalizes layer inputs to reduce internal covariate shift?",
      "options": ["Dropout", "Batch normalization", "Weight decay", "Momentum"],
      "answer": "Batch normalization"
    },
    {
      "question": "SC Q6: Which optimizer uses adaptive learning rates based on first and second moments of the gradients?",
      "options": ["SGD", "Adam", "RMSProp", "Momentum"],
      "answer": "Adam"
    },
    {
      "question": "SC Q7: Which of the following is NOT an advantage of dropout?",
      "options": ["Prevents overfitting", "Reduces model complexity", "Randomly disables neurons permanently", "Improves generalization"],
      "answer": "Randomly disables neurons permanently"
    },
    {
      "question": "SC Q8: Which loss function is most commonly paired with softmax for classification tasks?",
      "options": ["Mean Squared Error", "Hinge loss", "Cross-entropy loss", "L1 loss"],
      "answer": "Cross-entropy loss"
    },
    {
      "question": "SC Q9: Which activation function is zero-centered?",
      "options": ["ReLU", "Sigmoid", "Tanh", "Softmax"],
      "answer": "Tanh"
    },
    {
      "question": "SC Q10: Which activation function is defined as f(x) = max(0, x)?",
      "options": ["Sigmoid", "Tanh", "ReLU", "Softmax"],
      "answer": "ReLU"
    },
    {
      "question": "SC Q11: Which neural network architecture is primarily used for image processing?",
      "options": ["Recurrent Neural Network", "Convolutional Neural Network", "Feedforward Neural Network", "Transformer"],
      "answer": "Convolutional Neural Network"
    },
    {
      "question": "SC Q12: Which technique is used to reduce internal covariate shift in deep networks?",
      "options": ["Dropout", "Batch normalization", "Data augmentation", "Weight decay"],
      "answer": "Batch normalization"
    },
    {
      "question": "SC Q13: Which type of network is especially prone to vanishing gradients in deep configurations?",
      "options": ["CNN", "RNN", "MLP", "GAN"],
      "answer": "RNN"
    },
    {
      "question": "SC Q14: Which property describes the ReLU activation function?",
      "options": ["It outputs negative values for negative inputs", "It outputs zero for negative inputs", "It is zero-centered", "It saturates for large inputs"],
      "answer": "It outputs zero for negative inputs"
    },
    {
      "question": "SC Q15: Which method is used to visualize the contribution of input pixels to a model's prediction?",
      "options": ["Pooling", "Saliency maps", "Weight decay", "Dropout"],
      "answer": "Saliency maps"
    },
    {
      "question": "SC Q16: Which regularization method adds a penalty proportional to the square of the network weights?",
      "options": ["L1 regularization", "L2 regularization", "Dropout", "Data augmentation"],
      "answer": "L2 regularization"
    },
    {
      "question": "SC Q17: Which loss function is typically used with softmax for multi-class classification?",
      "options": ["Mean Squared Error", "Cross-entropy loss", "Hinge loss", "L1 loss"],
      "answer": "Cross-entropy loss"
    },
    {
      "question": "SC Q18: Which optimizer is known for combining momentum with adaptive learning rates?",
      "options": ["SGD", "Adam", "Adagrad", "RMSProp"],
      "answer": "Adam"
    },
    {
      "question": "SC Q19: Which generative model consists of a generator and a discriminator?",
      "options": ["Autoencoder", "Variational Autoencoder", "Generative Adversarial Network", "Recurrent Neural Network"],
      "answer": "Generative Adversarial Network"
    },
    {
      "question": "SC Q20: Which algorithm is designed for real-time object detection using a single forward pass?",
      "options": ["Faster R-CNN", "SSD", "YOLO", "RCNN"],
      "answer": "YOLO"
    },
    {
      "question": "SC Q21: Which attention mechanism uses fixed-size glimpses of an input?",
      "options": ["Soft attention", "Hard attention", "Self-attention", "Global attention"],
      "answer": "Hard attention"
    },
    {
      "question": "SC Q22: Which method is used to reduce the spatial dimensions of feature maps in CNNs?",
      "options": ["Convolution", "Pooling", "Normalization", "Activation"],
      "answer": "Pooling"
    },
    {
      "question": "SC Q23: Which approach learns features by reconstructing its input?",
      "options": ["Classifier", "Autoencoder", "Generative Adversarial Network", "Recurrent Neural Network"],
      "answer": "Autoencoder"
    },
    {
      "question": "SC Q24: Which statement best describes an autoencoder?",
      "options": ["A network that generates new data from noise", "A network that learns to reconstruct its input", "A network used for classification tasks", "A network that uses recurrent connections"],
      "answer": "A network that learns to reconstruct its input"
    },
    {
      "question": "SC Q25: Which network architecture is best suited for sequential data processing?",
      "options": ["Convolutional Neural Network", "Recurrent Neural Network", "Multilayer Perceptron", "Autoencoder"],
      "answer": "Recurrent Neural Network"
    },
    {
      "question": "SC Q26: Which recurrent unit includes gates to control information flow?",
      "options": ["Vanilla RNN", "LSTM", "MLP", "CNN"],
      "answer": "LSTM"
    },
    {
      "question": "SC Q27: Which recurrent neural network variant is simpler and uses fewer parameters than LSTM?",
      "options": ["GRU", "LSTM", "Bidirectional RNN", "Stacked RNN"],
      "answer": "GRU"
    },
    {
      "question": "SC Q28: Which method is used to train recurrent networks by unfolding them in time?",
      "options": ["Backpropagation", "Backpropagation Through Time", "Forward Propagation", "Stochastic Gradient Descent"],
      "answer": "Backpropagation Through Time"
    },
    {
      "question": "SC Q29: Which technique involves splitting long sequences into shorter segments for gradient computation?",
      "options": ["Full Backpropagation", "Truncated BPTT", "Mini-batch gradient descent", "Dropout"],
      "answer": "Truncated BPTT"
    },
    {
      "question": "SC Q30: Which metric is defined as the harmonic mean of precision and recall?",
      "options": ["Accuracy", "F1-Score", "ROC AUC", "Specificity"],
      "answer": "F1-Score"
    },
    {
      "question": "SC Q31: Which metric is particularly useful for evaluating imbalanced classification?",
      "options": ["Accuracy", "F1-Score", "Mean Squared Error", "Log Loss"],
      "answer": "F1-Score"
    },
    {
      "question": "SC Q32: Which method learns a latent representation by attempting to reproduce its input?",
      "options": ["Classifier", "Autoencoder", "Regressor", "Clustering"],
      "answer": "Autoencoder"
    },
    {
      "question": "SC Q33: Which technique projects high-dimensional data onto its principal components?",
      "options": ["t-SNE", "PCA", "Autoencoder", "LDA"],
      "answer": "PCA"
    },
    {
      "question": "SC Q34: Which approach uses a reparameterization trick to allow backpropagation through a stochastic layer?",
      "options": ["Autoencoder", "Variational Autoencoder", "GAN", "RNN"],
      "answer": "Variational Autoencoder"
    },
    {
      "question": "SC Q35: Which drawback is commonly associated with very deep neural networks?",
      "options": ["Overfitting", "Underfitting", "Vanishing gradients", "High bias"],
      "answer": "Vanishing gradients"
    },
    {
      "question": "SC Q36: Which technique improves gradient flow by introducing shortcut connections in deep networks?",
      "options": ["Dropout", "Residual connections", "Batch normalization", "Data augmentation"],
      "answer": "Residual connections"
    },
    {
      "question": "SC Q37: Which issue can occur during GAN training due to mode collapse?",
      "options": ["Overfitting", "Underfitting", "Mode collapse", "Vanishing gradients"],
      "answer": "Mode collapse"
    },
    {
      "question": "SC Q38: Which term describes using a pre-trained model on a new but related task?",
      "options": ["Transfer learning", "Multi-task learning", "Reinforcement learning", "Unsupervised learning"],
      "answer": "Transfer learning"
    },
    {
      "question": "SC Q39: Which technique involves applying a pre-trained network to a related problem?",
      "options": ["Transfer learning", "Domain adaptation", "Feature extraction", "Data augmentation"],
      "answer": "Transfer learning"
    },
    {
      "question": "SC Q40: Which term describes the simultaneous learning of multiple related tasks?",
      "options": ["Transfer learning", "Multi-task learning", "Ensemble learning", "Reinforcement learning"],
      "answer": "Multi-task learning"
    },
    {
      "question": "SC Q41: Which activation function is prone to saturation, potentially causing vanishing gradients?",
      "options": ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU"],
      "answer": "Sigmoid"
    },
    {
      "question": "SC Q42: Which technique helps mitigate the vanishing gradient problem in recurrent networks?",
      "options": ["Using sigmoid activations", "Using LSTM units", "Increasing learning rate", "Using dropout"],
      "answer": "Using LSTM units"
    },
    {
      "question": "SC Q43: Which optimizer is characterized by the use of momentum?",
      "options": ["SGD with momentum", "Adam", "RMSProp", "Adagrad"],
      "answer": "SGD with momentum"
    },
    {
      "question": "SC Q44: Which loss function is most commonly used to measure reconstruction error in autoencoders?",
      "options": ["Cross-entropy loss", "Mean Squared Error", "Hinge loss", "L1 loss"],
      "answer": "Mean Squared Error"
    },
    {
      "question": "SC Q45: Which regularization technique adds a penalty proportional to the absolute value of weights?",
      "options": ["L1 regularization", "L2 regularization", "Dropout", "Batch normalization"],
      "answer": "L1 regularization"
    },
    {
      "question": "SC Q46: Which pooling method selects the maximum value within a receptive field?",
      "options": ["Average pooling", "Max pooling", "Global pooling", "Stochastic pooling"],
      "answer": "Max pooling"
    },
    {
      "question": "SC Q47: Which pooling method computes the average value within a receptive field?",
      "options": ["Max pooling", "Average pooling", "Global pooling", "Stochastic pooling"],
      "answer": "Average pooling"
    },
    {
      "question": "SC Q48: Which layer type is used to reduce the number of parameters by combining features across channels?",
      "options": ["Convolutional layer", "1x1 Convolution", "Fully connected layer", "Pooling layer"],
      "answer": "1x1 Convolution"
    },
    {
      "question": "SC Q49: Which method in reinforcement learning aims to maximize the expected cumulative reward?",
      "options": ["Value iteration", "Policy optimization", "Q-learning", "Monte Carlo estimation"],
      "answer": "Policy optimization"
    },
    {
      "question": "SC Q50: Which strategy in reinforcement learning balances exploration and exploitation using probability distributions?",
      "options": ["Epsilon-greedy", "Softmax action selection", "Greedy action selection", "Deterministic policy"],
      "answer": "Softmax action selection"
    },
    {
      "question": "SC Q51: Which technique generates synthetic data using two competing networks?",
      "options": ["Autoencoder", "Variational Autoencoder", "Generative Adversarial Network", "Recurrent Neural Network"],
      "answer": "Generative Adversarial Network"
    },
    {
      "question": "SC Q52: Which module combines outputs from convolutional layers with different kernel sizes?",
      "options": ["Residual block", "Inception module", "Pooling layer", "Recurrent block"],
      "answer": "Inception module"
    },
    {
      "question": "SC Q53: Which architecture uses feature concatenation between encoder and decoder for segmentation?",
      "options": ["FCN", "U-Net", "ResNet", "AlexNet"],
      "answer": "U-Net"
    },
    {
      "question": "SC Q54: Which concept refers to halting training when validation performance degrades?",
      "options": ["Learning rate decay", "Early stopping", "Dropout", "Batch normalization"],
      "answer": "Early stopping"
    },
    {
      "question": "SC Q55: Which method involves gradually reducing the learning rate during training?",
      "options": ["Momentum", "Learning rate decay", "Dropout", "Data augmentation"],
      "answer": "Learning rate decay"
    },
    {
      "question": "SC Q56: Which term describes a model that performs well on training data but poorly on new data?",
      "options": ["Underfitting", "Overfitting", "Generalization", "Regularization"],
      "answer": "Overfitting"
    },
    {
      "question": "SC Q57: Which statistical test is commonly used to compare model performances?",
      "options": ["Chi-squared test", "Student's t-test", "ANOVA", "Wilcoxon test"],
      "answer": "Student's t-test"
    },
    {
      "question": "SC Q58: Which technique involves combining predictions from several models?",
      "options": ["Bagging", "Boosting", "Ensemble learning", "Cross-validation"],
      "answer": "Ensemble learning"
    },
    {
      "question": "SC Q59: Which metric measures the proportion of correct predictions made by a model?",
      "options": ["Accuracy", "Precision", "Recall", "F1-Score"],
      "answer": "Accuracy"
    },
    {
      "question": "SC Q60: Which metric is calculated as the ratio of true positives to the sum of true and false positives?",
      "options": ["Accuracy", "Precision", "Recall", "F1-Score"],
      "answer": "Precision"
    },
    {
      "question": "SC Q61: Which method is commonly used for semantic segmentation in medical imaging?",
      "options": ["Faster R-CNN", "Fully Convolutional Network", "YOLO", "R-CNN"],
      "answer": "Fully Convolutional Network"
    },
    {
      "question": "SC Q62: Which use case is a primary application of convolutional neural networks?",
      "options": ["Language translation", "Image classification", "Time series forecasting", "Graph processing"],
      "answer": "Image classification"
    },
    {
      "question": "SC Q63: Which approach is typically used to break symmetry during weight initialization?",
      "options": ["Zero initialization", "Random initialization", "Constant initialization", "Ones initialization"],
      "answer": "Random initialization"
    },
    {
      "question": "SC Q64: Which function is used to convert logits into probabilities?",
      "options": ["ReLU", "Softmax", "Tanh", "Sigmoid"],
      "answer": "Softmax"
    },
    {
      "question": "SC Q65: Which method uses past gradient information to accelerate convergence?",
      "options": ["Learning rate decay", "Momentum", "Dropout", "Batch normalization"],
      "answer": "Momentum"
    },
    {
      "question": "SC Q66: Which method is especially useful for sequential decision-making tasks in reinforcement learning?",
      "options": ["Q-learning", "Supervised learning", "Unsupervised learning", "Clustering"],
      "answer": "Q-learning"
    },
    {
      "question": "SC Q67: Which component of a GAN is trained to distinguish real data from generated data?",
      "options": ["Generator", "Encoder", "Discriminator", "Decoder"],
      "answer": "Discriminator"
    },
    {
      "question": "SC Q68: Which component of a GAN is responsible for generating realistic data samples?",
      "options": ["Discriminator", "Generator", "Encoder", "Classifier"],
      "answer": "Generator"
    },
    {
      "question": "SC Q69: Which method helps prevent overfitting by limiting model complexity?",
      "options": ["Regularization", "Data augmentation", "Increasing network size", "Early stopping"],
      "answer": "Regularization"
    },
    {
      "question": "SC Q70: Which optimizer adjusts learning rates individually based on squared gradients?",
      "options": ["SGD", "Adam", "AdaGrad", "Momentum"],
      "answer": "AdaGrad"
    },
    {
      "question": "SC Q71: Which technique normalizes activations across a mini-batch during training?",
      "options": ["Dropout", "Batch normalization", "Weight decay", "Data augmentation"],
      "answer": "Batch normalization"
    },
    {
      "question": "SC Q72: Which network architecture is best suited for processing sequential text data?",
      "options": ["CNN", "RNN", "MLP", "Transformer"],
      "answer": "RNN"
    },
    {
      "question": "SC Q73: Which method helps balance bias and variance in deep learning models?",
      "options": ["Regularization", "Data augmentation", "Increasing layers", "Dropout"],
      "answer": "Regularization"
    },
    {
      "question": "SC Q74: Which technique enforces sparsity in network weights?",
      "options": ["L2 regularization", "Batch normalization", "L1 regularization", "Dropout"],
      "answer": "L1 regularization"
    },
    {
      "question": "SC Q75: Which type of autoencoder is designed to produce a smooth latent space representation?",
      "options": ["Sparse autoencoder", "Denoising autoencoder", "Variational autoencoder", "Stacked autoencoder"],
      "answer": "Variational autoencoder"
    },
    {
      "question": "SC Q76: Which disadvantage is associated with fully connected layers in image processing?",
      "options": ["Low parameter count", "High computational cost", "Excellent spatial invariance", "Low memory requirements"],
      "answer": "High computational cost"
    },
    {
      "question": "SC Q77: Which approach is used to address class imbalance in training data?",
      "options": ["Under-sampling", "Over-sampling", "Data augmentation", "All of the above"],
      "answer": "All of the above"
    },
    {
      "question": "SC Q78: Which metric is particularly useful for evaluating performance on imbalanced datasets?",
      "options": ["Accuracy", "F1-Score", "Mean Squared Error", "Log Loss"],
      "answer": "F1-Score"
    },
    {
      "question": "SC Q79: Which method extracts important features by learning to reconstruct its input?",
      "options": ["Classifier", "Autoencoder", "Regressor", "Clustering"],
      "answer": "Autoencoder"
    },
    {
      "question": "SC Q80: Which technique uses a non-linear transformation to reduce data dimensionality?",
      "options": ["Linear Regression", "t-SNE", "PCA", "LDA"],
      "answer": "t-SNE"
    },
    {
      "question": "SC Q81: Which network is best suited for language modeling and translation?",
      "options": ["CNN", "RNN", "Autoencoder", "GAN"],
      "answer": "RNN"
    },
    {
      "question": "SC Q82: Which architecture incorporates self-attention mechanisms for capturing long-range dependencies?",
      "options": ["Transformer", "RNN", "CNN", "MLP"],
      "answer": "Transformer"
    },
    {
      "question": "SC Q83: Which unit calculates the output by summing weighted inputs and applying an activation function?",
      "options": ["Neuron", "Perceptron", "LSTM", "Convolution"],
      "answer": "Perceptron"
    },
    {
      "question": "SC Q84: Which strategy in reinforcement learning uses a target network updated less frequently?",
      "options": ["Fixed target network", "On-policy learning", "Off-policy learning", "Monte Carlo"],
      "answer": "Fixed target network"
    },
    {
      "question": "SC Q85: Which method computes the gradient of the loss with respect to the network weights?",
      "options": ["Forward propagation", "Backpropagation", "Gradient ascent", "Activation propagation"],
      "answer": "Backpropagation"
    },
    {
      "question": "SC Q86: Which method relies on the chain rule to compute gradients in neural networks?",
      "options": ["Forward propagation", "Backpropagation", "Dropout", "Normalization"],
      "answer": "Backpropagation"
    },
    {
      "question": "SC Q87: Which approach involves optimizing a min-max objective between two networks?",
      "options": ["Autoencoder", "Generative Adversarial Network", "Reinforcement Learning", "Clustering"],
      "answer": "Generative Adversarial Network"
    },
    {
      "question": "SC Q88: Which method in CNNs uses learned filters to extract local features?",
      "options": ["Pooling", "Convolution", "Normalization", "Activation"],
      "answer": "Convolution"
    },
    {
      "question": "SC Q89: Which pooling method is most commonly used to add non-linearity and reduce computational complexity?",
      "options": ["Average pooling", "Max pooling", "Sum pooling", "Stochastic pooling"],
      "answer": "Max pooling"
    },
    {
      "question": "SC Q90: Which strategy involves combining the predictions of multiple models to improve generalization?",
      "options": ["Regularization", "Ensemble learning", "Dropout", "Transfer learning"],
      "answer": "Ensemble learning"
    },
    {
      "question": "SC Q91: Which technique helps visualize the features learned by convolutional layers?",
      "options": ["Saliency maps", "Dropout", "Batch normalization", "Pooling"],
      "answer": "Saliency maps"
    },
    {
      "question": "SC Q92: Which dimensionality reduction technique preserves local structure in high-dimensional data?",
      "options": ["PCA", "t-SNE", "Autoencoder", "LDA"],
      "answer": "t-SNE"
    },
    {
      "question": "SC Q93: Which method is used to analyze the contribution of each neuron to the final prediction?",
      "options": ["Backpropagation", "Saliency maps", "Dropout", "Normalization"],
      "answer": "Saliency maps"
    },
    {
      "question": "SC Q94: Which technique is commonly used to initialize biases in neural networks?",
      "options": ["Random initialization", "Ones initialization", "Zero initialization", "Xavier initialization"],
      "answer": "Zero initialization"
    },
    {
      "question": "SC Q95: Which method adjusts network weights based on the error between predicted and actual outputs?",
      "options": ["Gradient descent", "Momentum", "Regularization", "Normalization"],
      "answer": "Gradient descent"
    },
    {
      "question": "SC Q96: Which challenge is commonly encountered when training very deep neural networks?",
      "options": ["Overfitting", "Underfitting", "Vanishing gradients", "High bias"],
      "answer": "Vanishing gradients"
    },
    {
      "question": "SC Q97: Which technique uses a gradient-based approach to optimize network weights?",
      "options": ["Stochastic Gradient Descent", "Genetic algorithms", "Random search", "Simulated annealing"],
      "answer": "Stochastic Gradient Descent"
    },
    {
      "question": "SC Q98: Which algorithm is commonly used for real-time object detection in images?",
      "options": ["Faster R-CNN", "YOLO", "SSD", "RCNN"],
      "answer": "YOLO"
    },
    {
      "question": "SC Q99: Which module in CNNs combines features from different receptive fields?",
      "options": ["Residual block", "Inception module", "Pooling layer", "Fully connected layer"],
      "answer": "Inception module"
    },
    {
      "question": "SC Q100: Which technique compresses the input data into a lower-dimensional representation?",
      "options": ["Clustering", "Autoencoder", "Regression", "Normalization"],
      "answer": "Autoencoder"
    }
  ],
  "multipleChoice": [
    {
      "question": "MC Q1: Which of the following are common activation functions in deep learning?",
      "options": ["ReLU", "Sigmoid", "Softmax", "Tanh"],
      "answer": ["ReLU", "Sigmoid", "Tanh"]
    },
    {
      "question": "MC Q2: Which of the following weight initialization methods are commonly used?",
      "options": ["Random initialization", "Xavier initialization", "He initialization", "Zero initialization"],
      "answer": ["Random initialization", "Xavier initialization", "He initialization"]
    },
    {
      "question": "MC Q3: Which of the following are considered regularization techniques?",
      "options": ["Dropout", "L1 regularization", "Batch normalization", "Data augmentation"],
      "answer": ["Dropout", "L1 regularization", "Data augmentation"]
    },
    {
      "question": "MC Q4: Which optimizers use adaptive learning rates?",
      "options": ["SGD", "Adam", "AdaGrad", "RMSProp"],
      "answer": ["Adam", "AdaGrad", "RMSProp"]
    },
    {
      "question": "MC Q5: Which evaluation metrics are commonly used for classification tasks?",
      "options": ["Accuracy", "Precision", "Recall", "Mean Squared Error"],
      "answer": ["Accuracy", "Precision", "Recall"]
    },
    {
      "question": "MC Q6: Which of the following architectures are examples of Convolutional Neural Networks?",
      "options": ["LeNet", "AlexNet", "VGG16", "ResNet"],
      "answer": ["LeNet", "AlexNet", "VGG16", "ResNet"]
    },
    {
      "question": "MC Q7: Which methods are used for dimensionality reduction?",
      "options": ["PCA", "t-SNE", "Autoencoder", "K-means clustering"],
      "answer": ["PCA", "t-SNE", "Autoencoder"]
    },
    {
      "question": "MC Q8: Which units are typically used in recurrent neural networks?",
      "options": ["Vanilla RNN", "LSTM", "GRU", "CNN"],
      "answer": ["Vanilla RNN", "LSTM", "GRU"]
    },
    {
      "question": "MC Q9: Which techniques help prevent overfitting in neural networks?",
      "options": ["Dropout", "Early stopping", "Data augmentation", "Increasing model complexity"],
      "answer": ["Dropout", "Early stopping", "Data augmentation"]
    },
    {
      "question": "MC Q10: Which evaluation metrics are used for image segmentation?",
      "options": ["Mean Intersection over Union", "Pixel Accuracy", "F1-Score", "Mean Squared Error"],
      "answer": ["Mean Intersection over Union", "Pixel Accuracy", "F1-Score"]
    },
    {
      "question": "MC Q11: Which techniques are used to visualize the features learned by CNNs?",
      "options": ["Saliency maps", "Guided backpropagation", "Occlusion", "Max pooling"],
      "answer": ["Saliency maps", "Guided backpropagation", "Occlusion"]
    },
    {
      "question": "MC Q12: Which of the following optimizers are used in deep learning?",
      "options": ["SGD", "Momentum", "Adam", "Dropout"],
      "answer": ["SGD", "Momentum", "Adam"]
    },
    {
      "question": "MC Q13: Which techniques are commonly used for data augmentation in image processing?",
      "options": ["Rotation", "Scaling", "Flipping", "Zero padding"],
      "answer": ["Rotation", "Scaling", "Flipping"]
    },
    {
      "question": "MC Q14: Which of the following are types of pooling methods?",
      "options": ["Max pooling", "Average pooling", "Sum pooling", "Global pooling"],
      "answer": ["Max pooling", "Average pooling", "Global pooling"]
    },
    {
      "question": "MC Q15: Which architectures incorporate skip connections?",
      "options": ["ResNet", "DenseNet", "VGG", "U-Net"],
      "answer": ["ResNet", "DenseNet", "U-Net"]
    },
    {
      "question": "MC Q16: Which methods are used in reinforcement learning?",
      "options": ["Q-learning", "Policy gradients", "Monte Carlo methods", "Principal Component Analysis"],
      "answer": ["Q-learning", "Policy gradients", "Monte Carlo methods"]
    },
    {
      "question": "MC Q17: Which loss functions are commonly used in deep learning?",
      "options": ["Cross-entropy", "Mean Squared Error", "Hinge loss", "Kullback-Leibler divergence"],
      "answer": ["Cross-entropy", "Mean Squared Error", "Hinge loss", "Kullback-Leibler divergence"]
    },
    {
      "question": "MC Q18: Which methods are used for feature extraction in image processing?",
      "options": ["Convolution", "Pooling", "Batch normalization", "Dropout"],
      "answer": ["Convolution", "Pooling", "Batch normalization"]
    },
    {
      "question": "MC Q19: Which techniques are effective in mitigating the vanishing gradient problem?",
      "options": ["ReLU", "Batch normalization", "Residual connections", "Sigmoid"],
      "answer": ["ReLU", "Batch normalization", "Residual connections"]
    },
    {
      "question": "MC Q20: Which regularization methods are used to reduce overfitting?",
      "options": ["L2 regularization", "L1 regularization", "Dropout", "Data augmentation"],
      "answer": ["L2 regularization", "L1 regularization", "Dropout", "Data augmentation"]
    },
    {
      "question": "MC Q21: Which techniques are used for transfer learning?",
      "options": ["Feature extraction", "Fine-tuning", "Random initialization", "Data augmentation"],
      "answer": ["Feature extraction", "Fine-tuning"]
    },
    {
      "question": "MC Q22: Which of the following are considered unsupervised learning methods?",
      "options": ["Clustering", "Autoencoders", "PCA", "Reinforcement Learning"],
      "answer": ["Clustering", "Autoencoders", "PCA"]
    },
    {
      "question": "MC Q23: Which methods are used to compute gradients in neural networks?",
      "options": ["Backpropagation", "Forward propagation", "Chain rule", "Dropout"],
      "answer": ["Backpropagation", "Chain rule"]
    },
    {
      "question": "MC Q24: Which techniques are used to improve the training stability of GANs?",
      "options": ["Feature matching", "Minibatch discrimination", "Unrolled GANs", "Dropout"],
      "answer": ["Feature matching", "Minibatch discrimination", "Unrolled GANs"]
    },
    {
      "question": "MC Q25: Which evaluation metrics are appropriate for regression tasks?",
      "options": ["Mean Squared Error", "Mean Absolute Error", "R-squared", "F1-Score"],
      "answer": ["Mean Squared Error", "Mean Absolute Error", "R-squared"]
    },
    {
      "question": "MC Q26: Which techniques are used to prevent overfitting besides dropout?",
      "options": ["Early stopping", "Data augmentation", "Ensemble learning", "Increasing network size"],
      "answer": ["Early stopping", "Data augmentation", "Ensemble learning"]
    },
    {
      "question": "MC Q27: Which methods are used to evaluate object detection performance?",
      "options": ["Intersection over Union", "Precision", "Recall", "F1-Score"],
      "answer": ["Intersection over Union", "Precision", "Recall", "F1-Score"]
    },
    {
      "question": "MC Q28: Which techniques are used for optimizing deep neural networks?",
      "options": ["Learning rate decay", "Momentum", "Adam", "Dropout"],
      "answer": ["Learning rate decay", "Momentum", "Adam"]
    },
    {
      "question": "MC Q29: Which approaches are used to initialize neural network weights?",
      "options": ["Random initialization", "Xavier initialization", "He initialization", "Zero initialization"],
      "answer": ["Random initialization", "Xavier initialization", "He initialization"]
    },
    {
      "question": "MC Q30: Which methods are effective for improving generalization in deep networks?",
      "options": ["Regularization", "Data augmentation", "Ensemble learning", "Increasing network depth"],
      "answer": ["Regularization", "Data augmentation", "Ensemble learning"]
    },
    {
      "question": "MC Q31: Which methods are used for compressing feature representations in CNNs?",
      "options": ["1x1 Convolutions", "Pooling", "Dropout", "Fully connected layers"],
      "answer": ["1x1 Convolutions", "Pooling"]
    },
    {
      "question": "MC Q32: Which techniques are used to measure the performance of clustering algorithms?",
      "options": ["Silhouette score", "Calinski-Harabasz index", "Accuracy", "Davies-Bouldin index"],
      "answer": ["Silhouette score", "Calinski-Harabasz index", "Davies-Bouldin index"]
    },
    {
      "question": "MC Q33: Which of the following are benefits of using transfer learning?",
      "options": ["Faster convergence", "Reduced need for large datasets", "Improved performance on related tasks", "Higher training time"],
      "answer": ["Faster convergence", "Reduced need for large datasets", "Improved performance on related tasks"]
    },
    {
      "question": "MC Q34: Which methods are used to extract temporal features from sequential data?",
      "options": ["Recurrent Neural Networks", "Convolutional Neural Networks", "LSTM", "GRU"],
      "answer": ["Recurrent Neural Networks", "LSTM", "GRU"]
    },
    {
      "question": "MC Q35: Which approaches are used to reduce the number of parameters in deep networks?",
      "options": ["Parameter sharing", "1x1 Convolutions", "Pruning", "Increasing network width"],
      "answer": ["Parameter sharing", "1x1 Convolutions", "Pruning"]
    },
    {
      "question": "MC Q36: Which techniques are used for model ensembling?",
      "options": ["Bagging", "Boosting", "Stacking", "Random initialization"],
      "answer": ["Bagging", "Boosting", "Stacking"]
    },
    {
      "question": "MC Q37: Which methods can be used to visualize decision boundaries in deep learning models?",
      "options": ["t-SNE", "PCA", "Saliency maps", "Activation maximization"],
      "answer": ["t-SNE", "PCA", "Saliency maps"]
    },
    {
      "question": "MC Q38: Which techniques are used to handle missing data in deep learning?",
      "options": ["Imputation", "Data augmentation", "Removing samples", "Feature scaling"],
      "answer": ["Imputation", "Removing samples"]
    },
    {
      "question": "MC Q39: Which approaches are used for unsupervised feature learning?",
      "options": ["Autoencoders", "Clustering", "PCA", "Supervised classification"],
      "answer": ["Autoencoders", "PCA"]
    },
    {
      "question": "MC Q40: Which techniques are used to calculate gradients in neural networks?",
      "options": ["Finite differences", "Backpropagation", "Automatic differentiation", "Random search"],
      "answer": ["Finite differences", "Backpropagation", "Automatic differentiation"]
    },
    {
      "question": "MC Q41: Which methods are used to regularize convolutional neural networks?",
      "options": ["Dropout", "Data augmentation", "Weight decay", "Increasing filter size"],
      "answer": ["Dropout", "Data augmentation", "Weight decay"]
    },
    {
      "question": "MC Q42: Which techniques are employed in reinforcement learning for exploration?",
      "options": ["Epsilon-greedy", "Softmax action selection", "Greedy selection", "Uniform random"],
      "answer": ["Epsilon-greedy", "Softmax action selection", "Uniform random"]
    },
    {
      "question": "MC Q43: Which loss functions are used in generative adversarial networks?",
      "options": ["Minimax loss", "Wasserstein loss", "Hinge loss", "Mean Squared Error"],
      "answer": ["Minimax loss", "Wasserstein loss", "Hinge loss"]
    },
    {
      "question": "MC Q44: Which methods are used for sequential data processing?",
      "options": ["Recurrent Neural Networks", "Convolutional Neural Networks", "Transformers", "Feedforward Neural Networks"],
      "answer": ["Recurrent Neural Networks", "Transformers"]
    },
    {
      "question": "MC Q45: Which techniques are used for learning feature hierarchies in images?",
      "options": ["Deep Convolutional Neural Networks", "Shallow networks", "Autoencoders", "Decision trees"],
      "answer": ["Deep Convolutional Neural Networks", "Autoencoders"]
    },
    {
      "question": "MC Q46: Which methods are commonly used for text representation in NLP?",
      "options": ["Word2Vec", "GloVe", "One-hot encoding", "Bag-of-words"],
      "answer": ["Word2Vec", "GloVe", "One-hot encoding", "Bag-of-words"]
    },
    {
      "question": "MC Q47: Which approaches are used to improve convergence speed in deep learning?",
      "options": ["Momentum", "Learning rate decay", "Batch normalization", "Increasing batch size"],
      "answer": ["Momentum", "Learning rate decay", "Batch normalization"]
    },
    {
      "question": "MC Q48: Which methods are used to combat the exploding gradient problem?",
      "options": ["Gradient clipping", "Increasing learning rate", "Using ReLU", "Batch normalization"],
      "answer": ["Gradient clipping", "Batch normalization"]
    },
    {
      "question": "MC Q49: Which techniques are used for object detection in deep learning?",
      "options": ["YOLO", "SSD", "Faster R-CNN", "K-means clustering"],
      "answer": ["YOLO", "SSD", "Faster R-CNN"]
    },
    {
      "question": "MC Q50: Which strategies are employed to fuse multi-scale features in CNNs?",
      "options": ["Inception modules", "Skip connections", "Pooling", "Fully connected layers"],
      "answer": ["Inception modules", "Skip connections"]
    },
    {
      "question": "MC Q51: Which methods are used to update neural network weights during training?",
      "options": ["Gradient descent", "Stochastic gradient descent", "Adam optimizer", "Genetic algorithms"],
      "answer": ["Gradient descent", "Stochastic gradient descent", "Adam optimizer"]
    },
    {
      "question": "MC Q52: Which techniques are common in unsupervised representation learning?",
      "options": ["Autoencoders", "Clustering", "Principal Component Analysis", "Supervised classification"],
      "answer": ["Autoencoders", "Principal Component Analysis"]
    },
    {
      "question": "MC Q53: Which methods are used to compute uncertainty in deep learning models?",
      "options": ["Bayesian neural networks", "Dropout", "Ensemble methods", "Deterministic networks"],
      "answer": ["Bayesian neural networks", "Dropout", "Ensemble methods"]
    },
    {
      "question": "MC Q54: Which techniques are applied to improve model interpretability?",
      "options": ["Saliency maps", "LIME", "SHAP", "Increasing model depth"],
      "answer": ["Saliency maps", "LIME", "SHAP"]
    },
    {
      "question": "MC Q55: Which methods are used to compute the receptive field in CNNs?",
      "options": ["Convolution operations", "Pooling layers", "Stride", "Padding"],
      "answer": ["Convolution operations", "Pooling layers", "Stride", "Padding"]
    },
    {
      "question": "MC Q56: Which approaches are used for learning sparse representations?",
      "options": ["L1 regularization", "Sparse autoencoders", "Dropout", "Data augmentation"],
      "answer": ["L1 regularization", "Sparse autoencoders"]
    },
    {
      "question": "MC Q57: Which of the following are advantages of using deep neural networks?",
      "options": ["Increased representational power", "Ability to learn hierarchical features", "Lower computational cost", "Improved feature reuse"],
      "answer": ["Increased representational power", "Ability to learn hierarchical features", "Improved feature reuse"]
    },
    {
      "question": "MC Q58: Which techniques are used for sequence-to-sequence modeling?",
      "options": ["Encoder-decoder architecture", "Attention mechanism", "Recurrent Neural Networks", "Convolutional Neural Networks"],
      "answer": ["Encoder-decoder architecture", "Attention mechanism", "Recurrent Neural Networks"]
    },
    {
      "question": "MC Q59: Which methods are used for visualizing high-dimensional data?",
      "options": ["t-SNE", "PCA", "MDS", "K-means clustering"],
      "answer": ["t-SNE", "PCA", "MDS"]
    },
    {
      "question": "MC Q60: Which methods are used to reduce the effect of overfitting by modifying the loss function?",
      "options": ["Weight decay", "Regularization terms", "Early stopping", "Increasing network depth"],
      "answer": ["Weight decay", "Regularization terms", "Early stopping"]
    },
    {
      "question": "MC Q61: Which approaches are used to measure model uncertainty in predictions?",
      "options": ["Bayesian inference", "Monte Carlo dropout", "Ensemble methods", "Deterministic prediction"],
      "answer": ["Bayesian inference", "Monte Carlo dropout", "Ensemble methods"]
    },
    {
      "question": "MC Q62: Which techniques are used to improve the robustness of neural networks?",
      "options": ["Data augmentation", "Adversarial training", "Batch normalization", "Overfitting"],
      "answer": ["Data augmentation", "Adversarial training", "Batch normalization"]
    },
    {
      "question": "MC Q63: Which methods are used for feature selection in deep learning?",
      "options": ["L1 regularization", "Dropout", "Principal Component Analysis", "Random forest feature importance"],
      "answer": ["L1 regularization", "Principal Component Analysis"]
    },
    {
      "question": "MC Q64: Which techniques are employed in self-supervised learning?",
      "options": ["Pretext tasks", "Contrastive learning", "Data augmentation", "Supervised labels"],
      "answer": ["Pretext tasks", "Contrastive learning", "Data augmentation"]
    },
    {
      "question": "MC Q65: Which methods are used for multi-label classification?",
      "options": ["Sigmoid activation", "Softmax activation", "Binary cross-entropy", "Multi-class cross-entropy"],
      "answer": ["Sigmoid activation", "Binary cross-entropy"]
    },
    {
      "question": "MC Q66: Which approaches are used to compress neural networks for deployment?",
      "options": ["Pruning", "Quantization", "Knowledge distillation", "Increasing model size"],
      "answer": ["Pruning", "Quantization", "Knowledge distillation"]
    },
    {
      "question": "MC Q67: Which techniques are applied to reduce the computational cost of CNNs?",
      "options": ["Depthwise separable convolutions", "1x1 Convolutions", "Pooling", "Increasing filter size"],
      "answer": ["Depthwise separable convolutions", "1x1 Convolutions", "Pooling"]
    },
    {
      "question": "MC Q68: Which methods are used to learn representations for graph-structured data?",
      "options": ["Graph Convolutional Networks", "Graph Attention Networks", "Recurrent Neural Networks", "Convolutional Neural Networks"],
      "answer": ["Graph Convolutional Networks", "Graph Attention Networks"]
    },
    {
      "question": "MC Q69: Which approaches are commonly used for time-series forecasting in deep learning?",
      "options": ["RNN", "LSTM", "GRU", "CNN"],
      "answer": ["RNN", "LSTM", "GRU"]
    },
    {
      "question": "MC Q70: Which techniques are used to compute attention weights in Transformers?",
      "options": ["Dot-product attention", "Scaled dot-product attention", "Additive attention", "Recurrent attention"],
      "answer": ["Dot-product attention", "Scaled dot-product attention", "Additive attention"]
    },
    {
      "question": "MC Q71: Which methods are used for learning from limited labeled data?",
      "options": ["Semi-supervised learning", "Transfer learning", "Data augmentation", "Full supervision"],
      "answer": ["Semi-supervised learning", "Transfer learning", "Data augmentation"]
    },
    {
      "question": "MC Q72: Which techniques are applied to accelerate training in deep learning?",
      "options": ["GPU acceleration", "Mixed precision training", "Distributed training", "Using CPU only"],
      "answer": ["GPU acceleration", "Mixed precision training", "Distributed training"]
    },
    {
      "question": "MC Q73: Which methods are used for hyperparameter optimization in deep learning?",
      "options": ["Grid search", "Random search", "Bayesian optimization", "Manual tuning"],
      "answer": ["Grid search", "Random search", "Bayesian optimization", "Manual tuning"]
    },
    {
      "question": "MC Q74: Which techniques are used to visualize the latent space of autoencoders?",
      "options": ["t-SNE", "PCA", "UMAP", "Confusion matrix"],
      "answer": ["t-SNE", "PCA", "UMAP"]
    },
    {
      "question": "MC Q75: Which methods are used to generate natural language from neural networks?",
      "options": ["Sequence-to-sequence models", "Transformers", "RNNs", "CNNs"],
      "answer": ["Sequence-to-sequence models", "Transformers", "RNNs"]
    },
    {
      "question": "MC Q76: Which techniques are used to handle long-term dependencies in RNNs?",
      "options": ["LSTM", "GRU", "Vanilla RNN", "Bidirectional RNN"],
      "answer": ["LSTM", "GRU", "Bidirectional RNN"]
    },
    {
      "question": "MC Q77: Which methods are used for learning distributed representations of words?",
      "options": ["Word2Vec", "GloVe", "FastText", "One-hot encoding"],
      "answer": ["Word2Vec", "GloVe", "FastText"]
    },
    {
      "question": "MC Q78: Which techniques are applied for adversarial training in deep learning?",
      "options": ["Generating adversarial examples", "Minibatch discrimination", "Defensive distillation", "Dropout"],
      "answer": ["Generating adversarial examples", "Defensive distillation"]
    },
    {
      "question": "MC Q79: Which methods are used to analyze the internal representations of deep networks?",
      "options": ["Activation maximization", "Deconvolution", "Guided backpropagation", "Pooling"],
      "answer": ["Activation maximization", "Deconvolution", "Guided backpropagation"]
    },
    {
      "question": "MC Q80: Which approaches are used for multi-modal learning?",
      "options": ["Late fusion", "Early fusion", "Hybrid fusion", "Single modality learning"],
      "answer": ["Late fusion", "Early fusion", "Hybrid fusion"]
    },
    {
      "question": "MC Q81: Which techniques are used to handle imbalanced datasets?",
      "options": ["Under-sampling", "Over-sampling", "Synthetic data generation", "Increasing model size"],
      "answer": ["Under-sampling", "Over-sampling", "Synthetic data generation"]
    },
    {
      "question": "MC Q82: Which methods are used to monitor and debug training in deep learning?",
      "options": ["TensorBoard", "Learning curves", "Activation histograms", "Confusion matrices"],
      "answer": ["TensorBoard", "Learning curves", "Activation histograms", "Confusion matrices"]
    },
    {
      "question": "MC Q83: Which approaches are used to combine features from multiple layers in a CNN?",
      "options": ["Skip connections", "Feature pyramid networks", "Concatenation", "Averaging"],
      "answer": ["Skip connections", "Feature pyramid networks", "Concatenation", "Averaging"]
    },
    {
      "question": "MC Q84: Which methods are used to compute the loss in object detection tasks?",
      "options": ["Cross-entropy loss", "Smooth L1 loss", "IoU loss", "Mean Squared Error"],
      "answer": ["Cross-entropy loss", "Smooth L1 loss", "IoU loss"]
    },
    {
      "question": "MC Q85: Which techniques are used for domain adaptation in deep learning?",
      "options": ["Fine-tuning", "Adversarial training", "Data augmentation", "Random initialization"],
      "answer": ["Fine-tuning", "Adversarial training", "Data augmentation"]
    },
    {
      "question": "MC Q86: Which methods are used to accelerate inference in deep learning models?",
      "options": ["Model pruning", "Quantization", "Knowledge distillation", "Increasing network depth"],
      "answer": ["Model pruning", "Quantization", "Knowledge distillation"]
    },
    {
      "question": "MC Q87: Which approaches are used to learn representations for video data?",
      "options": ["3D CNNs", "Recurrent Neural Networks", "Two-stream networks", "Autoencoders"],
      "answer": ["3D CNNs", "Recurrent Neural Networks", "Two-stream networks"]
    },
    {
      "question": "MC Q88: Which techniques are used for optimizing the architecture of neural networks?",
      "options": ["Neural Architecture Search", "Grid search", "Random search", "Manual tuning"],
      "answer": ["Neural Architecture Search", "Grid search", "Random search", "Manual tuning"]
    },
    {
      "question": "MC Q89: Which methods are used to evaluate the uncertainty of model predictions?",
      "options": ["Bayesian neural networks", "Monte Carlo dropout", "Ensemble methods", "Deterministic networks"],
      "answer": ["Bayesian neural networks", "Monte Carlo dropout", "Ensemble methods"]
    },
    {
      "question": "MC Q90: Which techniques are used for learning hierarchical representations?",
      "options": ["Deep Neural Networks", "Autoencoders", "Convolutional Neural Networks", "Decision trees"],
      "answer": ["Deep Neural Networks", "Autoencoders", "Convolutional Neural Networks"]
    },
    {
      "question": "MC Q91: Which methods are used to process graph-structured data?",
      "options": ["Graph Convolutional Networks", "Graph Attention Networks", "Recurrent Neural Networks", "CNNs"],
      "answer": ["Graph Convolutional Networks", "Graph Attention Networks"]
    },
    {
      "question": "MC Q92: Which approaches are used to incorporate temporal context in video analysis?",
      "options": ["3D CNNs", "RNNs", "Optical flow", "Temporal pooling"],
      "answer": ["3D CNNs", "RNNs", "Optical flow", "Temporal pooling"]
    },
    {
      "question": "MC Q93: Which methods are used to perform unsupervised clustering?",
      "options": ["K-means clustering", "Hierarchical clustering", "DBSCAN", "Supervised classification"],
      "answer": ["K-means clustering", "Hierarchical clustering", "DBSCAN"]
    },
    {
      "question": "MC Q94: Which techniques are used to reduce the dimensionality of text data?",
      "options": ["Latent Semantic Analysis", "Word2Vec", "TF-IDF", "Bag-of-words"],
      "answer": ["Latent Semantic Analysis", "TF-IDF", "Bag-of-words"]
    },
    {
      "question": "MC Q95: Which approaches are used for learning robust features against adversarial attacks?",
      "options": ["Adversarial training", "Defensive distillation", "Random noise injection", "Batch normalization"],
      "answer": ["Adversarial training", "Defensive distillation", "Random noise injection"]
    },
    {
      "question": "MC Q96: Which methods are used to integrate context information in segmentation tasks?",
      "options": ["Encoder-decoder architectures", "Skip connections", "Atrous convolutions", "Fully connected layers"],
      "answer": ["Encoder-decoder architectures", "Skip connections", "Atrous convolutions"]
    },
    {
      "question": "MC Q97: Which techniques are used to optimize deep reinforcement learning agents?",
      "options": ["Experience replay", "Target networks", "Policy gradients", "Supervised learning"],
      "answer": ["Experience replay", "Target networks", "Policy gradients"]
    },
    {
      "question": "MC Q98: Which methods are used for learning to generate images?",
      "options": ["Generative Adversarial Networks", "Variational Autoencoders", "Autoencoders", "Recurrent Neural Networks"],
      "answer": ["Generative Adversarial Networks", "Variational Autoencoders"]
    },
    {
      "question": "MC Q99: Which approaches are used for fine-tuning pre-trained models?",
      "options": ["Freezing layers", "Learning rate adjustment", "Data augmentation", "Random reinitialization"],
      "answer": ["Freezing layers", "Learning rate adjustment", "Data augmentation"]
    },
    {
      "question": "MC Q100: Which techniques are used to compress and encode image features into a lower-dimensional space?",
      "options": ["Autoencoders", "PCA", "t-SNE", "Clustering"],
      "answer": ["Autoencoders", "PCA", "t-SNE"]
    },
    {
      "question": "MC Q101: What is the correct order of steps to train a neural network using the backpropagation algorithm and an optimizer like Stochastic Gradient Descent (SGD)?",
      "options": [
        "D → B → C → A → E",
        "D → C → A → E → B",
        "D → A → C → E → B",
        "D → E → A → B → C",
        "D → B → E → C → A"
      ],
      "answer": ["D → C → A → E → B"]
    },
    {
      "question": "MC Q102: Which of the following statements are true regarding the Batch Normalization layer?",
      "options": [
        "It normalizes the distribution of the input for the layer which follows the Batch Normalization layer",
        "It normalizes the weights of each layer",
        "It is an effective way of back-propagation",
        "It can normalize the complete training dataset by computing the global mean and variance",
        "It has trainable parameters"
      ],
      "answer": [
        "It normalizes the distribution of the input for the layer which follows the Batch Normalization layer",
        "It can normalize the complete training dataset by computing the global mean and variance",
        "It has trainable parameters"
      ]
    },
    {
      "question": "MC Q103: Suppose you have a model trained on the ImageNet dataset for a classification task. Then you feed the model a blank image where every pixel is the same white color. For this input, the network will output the same score for each class. This statement is:",
      "options": [
        "True",
        "Cannot be answered",
        "False"
      ],
      "answer": ["True"]
    },
    {
      "question": "MC Q104: Which of the following functions do/does not fulfill the requirements for an activation function to train a deep learning model?",
      "options": [
        "tanh(x)",
        "1/x",
        "2x",
        "max(x, 0)",
        "1/(1+e^{-x})"
      ],
      "answer": ["1/x", "2x"]
    },
    {
      "question": "MC Q105: Which of the following statement(s) is/are wrong?",
      "options": [
        "The goal of a good policy is to maximize the future return",
        "The Bellman equations form a system of linear equations which can be solved for small problems",
        "Q Learning is an Off-policy method that does not need information about dynamics of the environment",
        "Temporal Difference Learning is an On-policy method that needs information about dynamics of the environment",
        "Greedy action selection policy is not always the best policy"
      ],
      "answer": ["Temporal Difference Learning is an On-policy method that needs information about dynamics of the environment"]
    },
    {
      "question": "MC Q106: Which of the following statements about the model capacity is/are true?",
      "options": [
        "The model capacity is determined by the number of training samples the network was optimized with",
        "The model capacity is linked to the variety of functions which can be approximated",
        "The model capacity is influenced by the depth of a network",
        "The tradeoff between bias and variance is closely related to the model capacity",
        "The bias in prediction necessarily increases when the model capacity increases as well"
      ],
      "answer": [
        "The model capacity is linked to the variety of functions which can be approximated",
        "The model capacity is influenced by the depth of a network",
        "The tradeoff between bias and variance is closely related to the model capacity"
      ]
    },
    {
      "question": "MC Q107: Which statement(s) is/are true about initializing your weights and biases?",
      "options": [
        "If the bias is initialized with 0, the gradient of the loss with respect to the bias is 0 as well",
        "Initializing your bias with 0 helps with the dying ReLU problem",
        "It is important to calibrate the variance of your weights",
        "Initialization does matter because the optimization of a Deep Learning model is a convex problem"
      ],
      "answer": ["It is important to calibrate the variance of your weights"]
    },
    {
      "question": "MC Q108: Which of the following statement is/are true regarding segmentation?",
      "options": [
        "The goal of segmentation is to draw a bounding box which contains the desired object",
        "Semantic segmentation can be considered as a pixel-wise classification",
        "A pixel-wise segmentation of an object can be converted into a bounding box",
        "Instance segmentation can only find one instance of a class per image"
      ],
      "answer": [
        "Semantic segmentation can be considered as a pixel-wise classification",
        "A pixel-wise segmentation of an object can be converted into a bounding box"
      ]
    },
    {
      "question": "MC Q101: Which of the following techniques is NOT a common practice to tackle class imbalance?",
      "options": ["Weighting loss with inverse class frequency", "Data augmentation", "Oversampling", "Batch normalization"],
      "answer": ["Batch normalization"]
    },
    {
      "question": "MC Q102: What is ensembling in the context of deep learning?",
      "options": ["A visualization technique with occlusion", "A network with multiple outputs", "Parallel paths with different kernel sizes", "Multiple networks with majority vote"],
      "answer": ["Multiple networks with majority vote"]
    },
    {
      "question": "MC Q103: Which option can be used to create synthetic data samples?",
      "options": ["The discriminator of a GAN", "The encoder of a Variational Autoencoder", "The decoder of a Variational Autoencoder", "The encoder of a GAN"],
      "answer": ["The decoder of a Variational Autoencoder"]
    },
    {
      "question": "MC Q104: Which of the following sampling strategies does NOT exist for sequence generation in RNNs?",
      "options": ["Greedy sampling", "Beam sampling", "Recognition sampling", "Random sampling"],
      "answer": ["Recognition sampling"]
    },
    {
      "question": "MC Q105: Which statement is false when implementing a Convolutional layer?",
      "options": ["The number of kernels determines the number of output channels", "Stride greater than one reduces the image size", "A 'valid' convolution outputs an image of the same size as its input", "The number of parameters in a kernel is independent of the image size"],
      "answer": ["A 'valid' convolution outputs an image of the same size as its input"]
    },
    {
      "question": "MC Q106: Why does initialization matter for the optimization of deep neural networks?",
      "options": ["Because the problem is non-convex", "To reduce the number of parameters", "To ensure convergence to the same local minimum", "To quickly find the steepest gradient"],
      "answer": ["Because the problem is non-convex"]
    },
    {
      "question": "MC Q107: What is the main task of Object Detection?",
      "options": ["Finding bounding boxes and classifying objects", "Searching for boundaries between objects", "Increasing image resolution", "Filling in missing image parts"],
      "answer": ["Finding bounding boxes and classifying objects"]
    },
    {
      "question": "MC Q108: Which statement about YOLO and Fast R-CNN is true?",
      "options": ["YOLO produces many object proposals for the CNN", "YOLO combines bounding box prediction and classification in one network", "Fast R-CNN is generally faster than YOLO", "Fast R-CNN can perform real-time detection"],
      "answer": ["YOLO combines bounding box prediction and classification in one network"]
    },
    {
      "question": "MC Q109: Which of the following examples is NOT considered a confound in dataset creation?",
      "options": ["Food images taken with different cameras", "Traffic images taken under different weather conditions", "Clothing images with gender-biased labels", "Language samples recorded with different microphones"],
      "answer": ["Clothing images with gender-biased labels"]
    },
    {
      "question": "MC Q110: Which method can be used to leverage a pre-trained model for a new image classification task?",
      "options": ["Freeze feature extraction layers and retrain classification layers", "Use the entire model and retrain the input layer", "Combine the original classification layer with a classical feature extractor", "Select and combine individual layers to form a new model"],
      "answer": ["Freeze feature extraction layers and retrain classification layers"]
    },
    {
      "question": "MC Q111: What is the correct order of steps to train a neural network using backpropagation and SGD?",
      "options": ["D → B → C → A → E", "D → C → A → E → B", "D → A → C → E → B", "D → E → A → B → C", "D → B → E → C → A"],
      "answer": ["D → C → A → E → B"]
    },
    {
      "question": "MC Q112: Which of the following statements about Batch Normalization are true?",
      "options": ["It normalizes the input distribution for the following layer", "It normalizes the weights of each layer", "It is an effective way of backpropagation", "It has trainable parameters", "It normalizes the entire training dataset using global statistics"],
      "answer": ["It normalizes the input distribution for the following layer", "It has trainable parameters"]
    },
    {
      "question": "MC Q113: When a model trained on ImageNet is fed a blank white image, what is the expected output?",
      "options": ["The network outputs the same score for each class", "The network outputs high confidence for one class", "The network outputs random scores", "The network fails to produce an output"],
      "answer": ["The network outputs the same score for each class"]
    },
    {
      "question": "MC Q114: Which of the following functions do NOT fulfill the requirements for an activation function?",
      "options": ["tanh(x)", "1/x", "2x", "max(x, 0)", "1/(1+e^{-x})"],
      "answer": ["1/x", "2x"]
    },
    {
      "question": "MC Q115: Which of the following statements is wrong regarding reinforcement learning methods?",
      "options": ["A good policy aims to maximize future return", "Bellman equations form a solvable system for small problems", "Q-Learning is off-policy and does not require environment dynamics", "Temporal Difference Learning is on-policy and requires environment dynamics", "Greedy action selection is always optimal"],
      "answer": ["Temporal Difference Learning is on-policy and requires environment dynamics", "Greedy action selection is always optimal"]
    },
    {
      "question": "MC Q116: Which of the following statements about model capacity are true?",
      "options": ["It is determined by the number of training samples", "It is linked to the variety of functions that can be approximated", "It is influenced by network depth", "It is unrelated to the bias-variance tradeoff", "The bias increases with increased model capacity"],
      "answer": ["It is linked to the variety of functions that can be approximated", "It is influenced by network depth"]
    },
    {
      "question": "MC Q117: Which of the following statements about weight and bias initialization are true?",
      "options": ["Initializing bias with 0 results in zero gradients for bias", "Zero bias initialization helps mitigate the dying ReLU problem", "It is important to calibrate the variance of weights", "Initialization does not affect optimization because the problem is convex"],
      "answer": ["It is important to calibrate the variance of weights"]
    },
    {
      "question": "MC Q118: Which of the following statements about segmentation are true?",
      "options": ["The goal of segmentation is to draw a bounding box", "Semantic segmentation is a pixel-wise classification", "A pixel-wise segmentation can be converted into a bounding box", "Instance segmentation can only detect one instance per image"],
      "answer": ["Semantic segmentation is a pixel-wise classification", "A pixel-wise segmentation can be converted into a bounding box"]
    },
    {
      "question": "MC Q119: What is the relationship between convolution and cross-correlation in deep learning?",
      "options": ["They are identical operations", "Convolution involves flipping the kernel while cross-correlation does not", "Cross-correlation is a type of convolution", "They are completely unrelated"],
      "answer": ["Convolution involves flipping the kernel while cross-correlation does not"]
    },
    {
      "question": "MC Q120: What does model capacity refer to in the context of neural networks?",
      "options": ["The number of parameters in the network", "The ability of a model to fit a wide variety of functions", "The depth of the network", "The computational cost during inference"],
      "answer": ["The ability of a model to fit a wide variety of functions"]
    },
    {
      "question": "MC Q121: Which of the following factors can lead to internal covariate shift in deep networks?",
      "options": ["Frequent updates of weights during training", "Changing distribution of layer inputs", "Poor weight initialization", "High dropout rates"],
      "answer": ["Changing distribution of layer inputs", "Poor weight initialization"]
    },
    {
      "question": "MC Q122: Why are Recurrent Neural Networks (RNNs) suitable for time-series problems?",
      "options": ["Because they process data sequentially", "Because they have memory to store previous inputs", "Because they ignore the order of data", "Because they use convolution operations"],
      "answer": ["Because they process data sequentially", "Because they have memory to store previous inputs"]
    },
    {
      "question": "MC Q123: What is the main benefit of using Long Short-Term Memory (LSTM) units over simple RNN cells?",
      "options": ["They are simpler to compute", "They mitigate the vanishing gradient problem", "They require fewer parameters", "They are faster to train"],
      "answer": ["They mitigate the vanishing gradient problem"]
    },
    {
      "question": "MC Q124: Which of the following evaluation metrics are specifically used for segmentation tasks?",
      "options": ["Intersection over Union (IoU)", "Dice coefficient", "Accuracy", "Precision"],
      "answer": ["Intersection over Union (IoU)", "Dice coefficient"]
    },
    {
      "question": "MC Q125: When visualizing the effects of a convolutional layer, what is typically visualized?",
      "options": ["The raw input image", "The feature maps (activation maps)", "The gradients of the layer", "The loss values"],
      "answer": ["The feature maps (activation maps)"]
    },
    {
      "question": "MC Q126: What is one primary benefit of applying a 1x1 convolution in a CNN?",
      "options": ["Dimensionality reduction", "Increased spatial resolution", "Non-linear activation", "Pooling of features"],
      "answer": ["Dimensionality reduction"]
    },
    {
      "question": "MC Q127: Why are ReLU activations often preferred over Sigmoid activations in fully connected layers?",
      "options": ["They help mitigate the vanishing gradient problem", "They are computationally less expensive", "They always produce outputs between 0 and 1", "They introduce non-linearity"],
      "answer": ["They help mitigate the vanishing gradient problem"]
    },
    {
      "question": "MC Q128: For an input image of size X×X, how many weights does a fully connected layer with Z output neurons require (ignoring biases)?",
      "options": ["X * X", "Z", "X * X * Z", "X + X + Z"],
      "answer": ["X * X * Z"]
    },
    {
      "question": "MC Q129: For a convolutional layer with N kernels of size K×K, how many weights are required per kernel (ignoring biases)?",
      "options": ["K * K", "N * K", "N * K * K", "K + K"],
      "answer": ["K * K"]
    },
    {
      "question": "MC Q130: What is one disadvantage of using fully connected layers for image classification?",
      "options": ["They ignore the spatial structure of images", "They use parameter sharing", "They require less memory", "They are less expressive"],
      "answer": ["They ignore the spatial structure of images"]
    },
    {
      "question": "MC Q131: What is one advantage of convolutional layers over fully connected layers?",
      "options": ["They exploit spatial hierarchies in data", "They require more parameters", "They ignore local features", "They are slower to compute"],
      "answer": ["They exploit spatial hierarchies in data"]
    },
    {
      "question": "MC Q132: Which of the following methods can be used to connect the output of a convolutional layer to a fully connected layer?",
      "options": ["Flattening", "Global average pooling", "Max pooling", "Both Flattening and Global average pooling"],
      "answer": ["Both Flattening and Global average pooling"]
    },
    {
      "question": "MC Q133: Why is it beneficial to learn a bias term in neural networks?",
      "options": ["It allows the activation function to shift", "It reduces overfitting", "It increases the number of parameters unnecessarily", "It speeds up training"],
      "answer": ["It allows the activation function to shift"]
    },
    {
      "question": "MC Q134: In multi-label classification problems, such as classifying multiple diseases, which activation function is more suitable?",
      "options": ["Sigmoid", "Softmax", "ReLU", "tanh"],
      "answer": ["Sigmoid"]
    },
    {
      "question": "MC Q135: What is the primary purpose of backpropagation in neural networks?",
      "options": ["To compute gradients for weight updates", "To initialize weights", "To perform forward propagation", "To calculate the loss"],
      "answer": ["To compute gradients for weight updates"]
    },
    {
      "question": "MC Q136: What is one key role of a skip connection in a residual network?",
      "options": ["To allow gradients to flow more easily", "To increase the number of layers", "To add non-linearity", "To reduce the model size"],
      "answer": ["To allow gradients to flow more easily"]
    },
    {
      "question": "MC Q137: In reinforcement learning, what constitutes an 'action'?",
      "options": ["A decision taken by the agent", "A reward signal", "The current state", "A parameter update"],
      "answer": ["A decision taken by the agent"]
    },
    {
      "question": "MC Q138: In reinforcement learning, what is a 'reward'?",
      "options": ["A numerical feedback signal", "A change in the agent's state", "The policy used by the agent", "A measure of exploration"],
      "answer": ["A numerical feedback signal"]
    },
    {
      "question": "MC Q139: What is a primary limitation of a greedy action selection algorithm?",
      "options": ["It only considers immediate rewards", "It is computationally intensive", "It always chooses random actions", "It explores the environment sufficiently"],
      "answer": ["It only considers immediate rewards"]
    },
    {
      "question": "MC Q140: What does the epsilon greedy policy balance in reinforcement learning?",
      "options": ["Exploration and exploitation", "Training speed and accuracy", "Memory usage and performance", "Model complexity and interpretability"],
      "answer": ["Exploration and exploitation"]
    },
    {
      "question": "MC Q141: What does a Markov Decision Process (MDP) model in reinforcement learning?",
      "options": ["The agent's actions and rewards", "The state transitions and rewards in the environment", "Only the agent's policy", "Only the environmental dynamics"],
      "answer": ["The state transitions and rewards in the environment"]
    },
    {
      "question": "MC Q142: What is the main idea behind the Nesterov Accelerated Gradient (NAG) optimizer?",
      "options": ["It looks ahead to compute the gradient", "It uses momentum to accelerate convergence", "It adapts the learning rate based on past gradients", "It uses second-order derivatives"],
      "answer": ["It looks ahead to compute the gradient", "It uses momentum to accelerate convergence"]
    },
    {
      "question": "MC Q143: What is the purpose of using a ConvTranspose2d layer in a UNet architecture?",
      "options": ["To perform upsampling", "To perform downsampling", "To reduce the number of channels", "To increase the receptive field"],
      "answer": ["To perform upsampling"]
    },
    {
      "question": "MC Q144: What is the benefit of concatenating feature maps in a UNet architecture?",
      "options": ["It merges encoder and decoder features", "It increases the number of parameters", "It reduces the spatial dimensions", "It simplifies the network structure"],
      "answer": ["It merges encoder and decoder features"]
    },
    {
      "question": "MC Q145: Which activation function is typically used in the output layer for binary classification tasks?",
      "options": ["Sigmoid", "Softmax", "ReLU", "tanh"],
      "answer": ["Sigmoid"]
    },
    {
      "question": "MC Q146: Which loss function is most suitable for binary classification problems?",
      "options": ["Binary Cross Entropy Loss", "Mean Squared Error Loss", "Categorical Cross Entropy Loss", "Hinge Loss"],
      "answer": ["Binary Cross Entropy Loss"]
    },
    {
      "question": "MC Q147: How is accuracy computed from a confusion matrix?",
      "options": ["(True Positives + True Negatives) / Total Samples", "True Positives / Total Samples", "False Positives / Total Samples", "True Negatives / Total Samples"],
      "answer": ["(True Positives + True Negatives) / Total Samples"]
    },
    {
      "question": "MC Q148: What is a potential concern when evaluating model performance on a highly imbalanced dataset?",
      "options": ["High accuracy may be misleading", "The model may overfit to the majority class", "Precision and recall may be low", "The confusion matrix becomes irrelevant"],
      "answer": ["High accuracy may be misleading", "The model may overfit to the majority class"]
    },
    {
      "question": "MC Q149: Why might a high accuracy not be sufficient to conclude that a classification problem is solved?",
      "options": ["Because of class imbalance", "Because of overfitting", "Because accuracy doesn't measure recall", "Because accuracy is always a reliable metric"],
      "answer": ["Because of class imbalance", "Because of overfitting"]
    },
    {
      "question": "MC Q150: What is the role of the bias term in a neural network?",
      "options": ["To allow the activation function to shift", "To normalize the input", "To reduce the number of parameters", "To control the learning rate"],
      "answer": ["To allow the activation function to shift"]
    },
    {
      "question": "MC Q151: Which activation function is used in the network architecture described in the SS2021 backpropagation question?",
      "options": ["Sigmoid", "ReLU", "tanh", "Softmax"],
      "answer": ["Sigmoid"]
    },
    {
      "question": "MC Q152: What loss function is used in the network described in the SS2021 backpropagation question?",
      "options": ["L2 norm (Mean Squared Error)", "Cross Entropy Loss", "Hinge Loss", "L1 norm"],
      "answer": ["L2 norm (Mean Squared Error)"]
    },
    {
      "question": "MC Q153: What is the general purpose of regularization in deep learning?",
      "options": ["To prevent overfitting", "To increase training speed", "To improve model interpretability", "To reduce underfitting"],
      "answer": ["To prevent overfitting"]
    },
    {
      "question": "MC Q154: Which of the following norms are commonly used for weight regularization in deep learning?",
      "options": ["L1 norm", "L2 norm", "L∞ norm", "Frobenius norm"],
      "answer": ["L1 norm", "L2 norm"]
    },
    {
      "question": "MC Q155: Which layer is typically added to mitigate internal covariate shift in neural networks?",
      "options": ["Batch Normalization layer", "Dropout layer", "Pooling layer", "Activation layer"],
      "answer": ["Batch Normalization layer"]
    },
    {
      "question": "MC Q156: What is the primary purpose of generating saliency maps in deep learning?",
      "options": ["To visualize important regions in the input", "To reduce the model size", "To augment the dataset", "To initialize weights"],
      "answer": ["To visualize important regions in the input"]
    },
    {
      "question": "MC Q157: What does multi-task learning in neural networks refer to?",
      "options": ["Training on additional related tasks to improve the main task", "Using multiple optimizers simultaneously", "Ensembling different models", "Regularizing the network using dropout"],
      "answer": ["Training on additional related tasks to improve the main task"]
    },
    {
      "question": "MC Q158: What issue does the Exponential Linear Unit (ELU) activation function aim to address?",
      "options": ["Vanishing gradients", "Overfitting", "Exploding gradients", "Underfitting"],
      "answer": ["Vanishing gradients"]
    },
    {
      "question": "MC Q159: What is the effect of increasing the padding in a convolutional layer in PyTorch?",
      "options": ["It preserves spatial dimensions", "It reduces the output size", "It increases the number of parameters", "It decreases computational cost"],
      "answer": ["It preserves spatial dimensions"]
    },
    {
      "question": "MC Q160: What is the primary function of a fully connected (Linear) layer in a Convolutional Neural Network?",
      "options": ["To aggregate features and perform classification", "To perform convolution operations", "To reduce overfitting", "To perform pooling"],
      "answer": ["To aggregate features and perform classification"]
    },
    {
      "question": "MC Q161: Which activation function is most commonly used in modern deep neural networks for hidden layers?",
      "options": ["ReLU", "Sigmoid", "tanh", "Softmax"],
      "answer": ["ReLU"]
    },
    {
      "question": "MC Q162: What is the main disadvantage of using Sigmoid activation in deep networks?",
      "options": ["Vanishing gradient problem", "Exploding gradient problem", "Excessive sparsity", "Non-differentiability"],
      "answer": ["Vanishing gradient problem"]
    },
    {
      "question": "MC Q163: What does dropout regularization do in a neural network?",
      "options": ["Randomly drops units during training", "Increases the number of neurons", "Boosts the learning rate", "Normalizes the input data"],
      "answer": ["Randomly drops units during training"]
    },
    {
      "question": "MC Q164: Which optimizer adapts the learning rate for each parameter based on estimates of first and second moments of the gradients?",
      "options": ["Adam", "SGD", "NAG", "RMSProp"],
      "answer": ["Adam"]
    },
    {
      "question": "MC Q165: What is the primary role of the learning rate in gradient descent optimization?",
      "options": ["To determine the step size during weight updates", "To control the momentum", "To regularize the model", "To normalize the gradients"],
      "answer": ["To determine the step size during weight updates"]
    },
    {
      "question": "MC Q166: Which of the following are common methods for weight initialization?",
      "options": ["Xavier/Glorot initialization", "He initialization", "Random initialization", "Zero initialization"],
      "answer": ["Xavier/Glorot initialization", "He initialization", "Random initialization"]
    },
    {
      "question": "MC Q167: What does 'overfitting' refer to in machine learning?",
      "options": ["Model performs well on training data but poorly on unseen data", "Model performs poorly on both training and test data", "Model has too few parameters", "Model is too simple"],
      "answer": ["Model performs well on training data but poorly on unseen data"]
    },
    {
      "question": "MC Q168: Which techniques can be used to reduce overfitting in deep neural networks?",
      "options": ["Dropout", "Early stopping", "Data augmentation", "Increasing model complexity"],
      "answer": ["Dropout", "Early stopping", "Data augmentation"]
    },
    {
      "question": "MC Q169: What is transfer learning in deep learning?",
      "options": ["Using a pre-trained model on a new, related task", "Training a model from scratch", "Ensembling multiple models", "Using data augmentation techniques"],
      "answer": ["Using a pre-trained model on a new, related task"]
    },
    {
      "question": "MC Q170: What is the primary purpose of data augmentation in training deep neural networks?",
      "options": ["To artificially expand the training dataset", "To reduce the model complexity", "To improve computational efficiency", "To prevent overfitting"],
      "answer": ["To artificially expand the training dataset", "To prevent overfitting"]
    },
    {
      "question": "MC Q171: Which of the following are characteristic features of convolutional neural networks (CNNs)?",
      "options": ["Local connectivity", "Weight sharing", "Fully connected layers", "Invariance to translation"],
      "answer": ["Local connectivity", "Weight sharing", "Invariance to translation"]
    },
    {
      "question": "MC Q172: What is the role of pooling layers in CNNs?",
      "options": ["To reduce the spatial dimensions", "To introduce non-linearity", "To combine features", "To prevent overfitting"],
      "answer": ["To reduce the spatial dimensions", "To combine features"]
    },
    {
      "question": "MC Q173: Which of the following are common types of pooling operations in CNNs?",
      "options": ["Max pooling", "Average pooling", "Sum pooling", "Min pooling"],
      "answer": ["Max pooling", "Average pooling"]
    },
    {
      "question": "MC Q174: What is the key idea behind residual networks (ResNets)?",
      "options": ["Adding shortcut connections", "Using deeper architectures", "Employing dropout", "Using batch normalization"],
      "answer": ["Adding shortcut connections"]
    },
    {
      "question": "MC Q175: What does 'batch size' refer to in the context of training neural networks?",
      "options": ["The number of training examples used in one iteration", "The total number of training samples", "The number of layers in the network", "The number of parameters in the model"],
      "answer": ["The number of training examples used in one iteration"]
    },
    {
      "question": "MC Q176: Which of the following are benefits of using GPUs for deep learning?",
      "options": ["Parallel processing capabilities", "Faster matrix computations", "Lower energy consumption", "Optimized for deep learning frameworks"],
      "answer": ["Parallel processing capabilities", "Faster matrix computations", "Optimized for deep learning frameworks"]
    },
    {
      "question": "MC Q177: What is a hyperparameter in machine learning?",
      "options": ["A parameter set before training", "A parameter learned during training", "The weights of the neural network", "The bias term in the network"],
      "answer": ["A parameter set before training"]
    },
    {
      "question": "MC Q178: Which of the following are examples of hyperparameters in deep learning models?",
      "options": ["Learning rate", "Batch size", "Number of epochs", "Weight values"],
      "answer": ["Learning rate", "Batch size", "Number of epochs"]
    },
    {
      "question": "MC Q179: What does 'gradient descent' refer to in optimization?",
      "options": ["An algorithm to minimize the loss function", "A method to compute gradients", "A technique for weight initialization", "A strategy for data augmentation"],
      "answer": ["An algorithm to minimize the loss function"]
    },
    {
      "question": "MC Q180: Which of the following are variations of gradient descent?",
      "options": ["Stochastic Gradient Descent (SGD)", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Newton's Method"],
      "answer": ["Stochastic Gradient Descent (SGD)", "Mini-batch Gradient Descent", "Batch Gradient Descent"]
    },
    {
      "question": "MC Q181: What is the purpose of using momentum in gradient descent optimization?",
      "options": ["To accelerate convergence", "To escape local minima", "To smooth out oscillations", "To adapt the learning rate"],
      "answer": ["To accelerate convergence", "To smooth out oscillations"]
    },
    {
      "question": "MC Q182: Which of the following methods can help in escaping local minima during training?",
      "options": ["Momentum", "Learning rate scheduling", "Dropout", "Random initialization"],
      "answer": ["Momentum", "Learning rate scheduling"]
    },
    {
      "question": "MC Q183: What does the term 'epoch' mean in the context of training machine learning models?",
      "options": ["One complete pass through the entire training dataset", "A single weight update", "A batch of training samples", "A measure of model performance"],
      "answer": ["One complete pass through the entire training dataset"]
    },
    {
      "question": "MC Q184: Which of the following techniques are commonly used for model evaluation?",
      "options": ["Cross-validation", "Hold-out validation", "Bootstrapping", "Gradient descent"],
      "answer": ["Cross-validation", "Hold-out validation", "Bootstrapping"]
    },
    {
      "question": "MC Q185: What is the purpose of a validation set in machine learning?",
      "options": ["To tune hyperparameters", "To train the model", "To evaluate model performance during training", "To prevent overfitting"],
      "answer": ["To tune hyperparameters", "To evaluate model performance during training"]
    },
    {
      "question": "MC Q186: What is transfer learning commonly used for in deep learning?",
      "options": ["Leveraging pre-trained models for new tasks", "Training models faster", "Reducing model size", "Improving data quality"],
      "answer": ["Leveraging pre-trained models for new tasks"]
    },
    {
      "question": "MC Q187: Which of the following are common challenges in training deep neural networks?",
      "options": ["Vanishing gradients", "Overfitting", "Insufficient data", "Model interpretability"],
      "answer": ["Vanishing gradients", "Overfitting", "Insufficient data"]
    },
    {
      "question": "MC Q188: What is the role of the activation function in a neural network?",
      "options": ["To introduce non-linearity", "To compute gradients", "To update weights", "To normalize inputs"],
      "answer": ["To introduce non-linearity"]
    },
    {
      "question": "MC Q189: Which of the following metrics are commonly used to evaluate classification models?",
      "options": ["Accuracy", "Precision", "Recall", "Mean Squared Error"],
      "answer": ["Accuracy", "Precision", "Recall"]
    },
    {
      "question": "MC Q190: What does the F1 score represent in classification tasks?",
      "options": ["The harmonic mean of precision and recall", "The arithmetic mean of precision and recall", "The difference between precision and recall", "The product of precision and recall"],
      "answer": ["The harmonic mean of precision and recall"]
    },
    {
      "question": "MC Q191: Which of the following are types of loss functions used in deep learning?",
      "options": ["Mean Squared Error", "Cross Entropy Loss", "Hinge Loss", "Cosine Similarity Loss"],
      "answer": ["Mean Squared Error", "Cross Entropy Loss", "Hinge Loss"]
    },
    {
      "question": "MC Q192: What is the significance of using a non-linear activation function in hidden layers?",
      "options": ["It allows the network to learn complex functions", "It reduces the number of parameters", "It makes the network linear", "It speeds up convergence"],
      "answer": ["It allows the network to learn complex functions"]
    },
    {
      "question": "MC Q193: Which of the following practices can help improve the generalization of a deep learning model?",
      "options": ["Regularization", "Data augmentation", "Early stopping", "Increasing model capacity"],
      "answer": ["Regularization", "Data augmentation", "Early stopping"]
    },
    {
      "question": "MC Q194: What does the term 'backpropagation' refer to in neural networks?",
      "options": ["The process of propagating the error gradient backwards", "The forward pass computation", "Weight initialization", "Data preprocessing"],
      "answer": ["The process of propagating the error gradient backwards"]
    },
    {
      "question": "MC Q195: Which of the following are advantages of using deep learning over traditional machine learning methods?",
      "options": ["Automatic feature extraction", "Scalability to large datasets", "High interpretability", "State-of-the-art performance in many tasks"],
      "answer": ["Automatic feature extraction", "Scalability to large datasets", "State-of-the-art performance in many tasks"]
    },
    {
      "question": "MC Q196: What is the primary challenge of training very deep neural networks?",
      "options": ["Vanishing gradients", "Excessive computational cost", "Lack of data", "Simple optimization"],
      "answer": ["Vanishing gradients"]
    },
    {
      "question": "MC Q197: Which of the following factors can affect the convergence of a deep neural network?",
      "options": ["Learning rate", "Weight initialization", "Batch size", "Activation function"],
      "answer": ["Learning rate", "Weight initialization", "Batch size", "Activation function"]
    },
    {
      "question": "MC Q198: What is the role of the optimizer in training a neural network?",
      "options": ["To update the weights based on computed gradients", "To compute the loss", "To initialize the network", "To perform the forward pass"],
      "answer": ["To update the weights based on computed gradients"]
    },
    {
      "question": "MC Q199: Which of the following are common frameworks used for building deep learning models?",
      "options": ["TensorFlow", "PyTorch", "Scikit-learn", "Keras"],
      "answer": ["TensorFlow", "PyTorch", "Keras"]
    },
    {
      "question": "MC Q200: What is the primary goal of using deep learning in computer vision tasks?",
      "options": ["To automatically learn hierarchical features from data", "To manually design features", "To reduce computational complexity", "To apply traditional machine learning methods"],
      "answer": ["To automatically learn hierarchical features from data"]
    }    

  ],
  "shortAnswer": [
    {
      "question": "SA Q1: What does CNN stand for in deep learning?",
      "answer": "Convolutional Neural Network"
    },
    {
      "question": "SA Q2: What does RNN stand for?",
      "answer": "Recurrent Neural Network"
    },
    {
      "question": "SA Q3: What is the primary purpose of dropout in neural networks?",
      "answer": "To prevent overfitting by randomly deactivating neurons during training."
    },
    {
      "question": "SA Q4: Define softmax in the context of neural networks.",
      "answer": "Softmax is a function that converts raw output scores (logits) into probabilities that sum to one."
    },
    {
      "question": "SA Q5: What is overfitting in machine learning?",
      "answer": "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on unseen data."
    },
    {
      "question": "SA Q6: What is an autoencoder?",
      "answer": "An autoencoder is a neural network that learns to compress input data into a lower-dimensional representation and then reconstruct it."
    },
    {
      "question": "SA Q7: What is the purpose of batch normalization?",
      "answer": "Batch normalization normalizes layer inputs to stabilize and speed up training."
    },
    {
      "question": "SA Q8: Define the term 'vanishing gradient'.",
      "answer": "Vanishing gradient is a problem where gradients become very small, hindering weight updates in deep networks."
    },
    {
      "question": "SA Q9: What is a Generative Adversarial Network (GAN)?",
      "answer": "A GAN consists of two networks—a generator and a discriminator—that compete to produce realistic synthetic data."
    },
    {
      "question": "SA Q10: What is the role of the discriminator in a GAN?",
      "answer": "The discriminator distinguishes between real data samples and those generated by the generator."
    },
    {
      "question": "SA Q11: What does the term 'transfer learning' mean?",
      "answer": "Transfer learning involves using a pre-trained model on a new, related task to improve learning efficiency."
    },
    {
      "question": "SA Q12: What is meant by 'fine-tuning' in the context of deep learning?",
      "answer": "Fine-tuning is the process of adjusting the parameters of a pre-trained model on a new task."
    },
    {
      "question": "SA Q13: What is the purpose of the ReLU activation function?",
      "answer": "ReLU introduces non-linearity and helps mitigate the vanishing gradient problem by outputting zero for negative inputs."
    },
    {
      "question": "SA Q14: What is meant by 'model capacity'?",
      "answer": "Model capacity refers to the range and complexity of functions a model can approximate."
    },
    {
      "question": "SA Q15: Explain the concept of 'learning rate decay'.",
      "answer": "Learning rate decay gradually reduces the learning rate during training to help the model converge smoothly."
    },
    {
      "question": "SA Q16: What is the F1-Score?",
      "answer": "The F1-Score is the harmonic mean of precision and recall, used to evaluate classification performance."
    },
    {
      "question": "SA Q17: What is the main purpose of a convolutional layer in CNNs?",
      "answer": "A convolutional layer extracts local features from input images using learnable filters."
    },
    {
      "question": "SA Q18: What does 'backpropagation' refer to in neural networks?",
      "answer": "Backpropagation is the process of computing gradients and updating network weights by propagating errors backward."
    },
    {
      "question": "SA Q19: Define 'gradient descent'.",
      "answer": "Gradient descent is an optimization algorithm that adjusts weights to minimize the loss function."
    },
    {
      "question": "SA Q20: What is a recurrent neural network used for?",
      "answer": "RNNs are used for processing sequential data by maintaining a hidden state across time steps."
    },
    {
      "question": "SA Q21: What is the purpose of the LSTM unit in RNNs?",
      "answer": "LSTM units help capture long-term dependencies by using gating mechanisms to control information flow."
    },
    {
      "question": "SA Q22: What is a GRU and how does it differ from an LSTM?",
      "answer": "A GRU is a simpler variant of LSTM that combines the forget and input gates into a single update gate."
    },
    {
      "question": "SA Q23: What is meant by 'dropout rate'?",
      "answer": "Dropout rate is the probability with which neurons are randomly deactivated during training."
    },
    {
      "question": "SA Q24: What is 'batch size' in neural network training?",
      "answer": "Batch size is the number of samples processed before the model’s parameters are updated."
    },
    {
      "question": "SA Q25: What does PCA stand for?",
      "answer": "Principal Component Analysis"
    },
    {
      "question": "SA Q26: How does PCA reduce data dimensionality?",
      "answer": "PCA projects data onto the directions (principal components) that maximize variance."
    },
    {
      "question": "SA Q27: What is t-SNE used for?",
      "answer": "t-SNE is used for visualizing high-dimensional data by reducing it to two or three dimensions."
    },
    {
      "question": "SA Q28: What is the purpose of a loss function in neural networks?",
      "answer": "The loss function measures the difference between predicted outputs and true values, guiding weight updates."
    },
    {
      "question": "SA Q29: Define 'regularization' in the context of deep learning.",
      "answer": "Regularization is a technique used to reduce overfitting by penalizing model complexity."
    },
    {
      "question": "SA Q30: What is meant by 'overparameterization'?",
      "answer": "Overparameterization refers to having more model parameters than necessary, which can lead to overfitting."
    },
    {
      "question": "SA Q31: What is meant by 'feature extraction'?",
      "answer": "Feature extraction is the process of transforming raw data into a set of features that can be effectively used for a task."
    },
    {
      "question": "SA Q32: What is an embedding in deep learning?",
      "answer": "An embedding is a dense vector representation of discrete items, such as words or images."
    },
    {
      "question": "SA Q33: What does the term 'activation function' mean?",
      "answer": "An activation function introduces non-linearity into a neural network, allowing it to learn complex patterns."
    },
    {
      "question": "SA Q34: What is meant by 'loss convergence'?",
      "answer": "Loss convergence occurs when the loss function reaches a stable minimum value during training."
    },
    {
      "question": "SA Q35: What is the purpose of using an optimizer in deep learning?",
      "answer": "An optimizer adjusts the model's weights to minimize the loss function."
    },
    {
      "question": "SA Q36: What does 'epoch' refer to in the training process?",
      "answer": "An epoch is one complete pass through the entire training dataset."
    },
    {
      "question": "SA Q37: What is 'early stopping'?",
      "answer": "Early stopping is a technique that stops training when the validation performance begins to deteriorate."
    },
    {
      "question": "SA Q38: What does 'fine-tuning' mean in transfer learning?",
      "answer": "Fine-tuning involves adjusting a pre-trained model's parameters on a new dataset."
    },
    {
      "question": "SA Q39: What is the main idea behind ensemble learning?",
      "answer": "Ensemble learning combines the predictions of multiple models to improve overall performance."
    },
    {
      "question": "SA Q40: Define 'learning rate' in the context of neural network optimization.",
      "answer": "The learning rate is a hyperparameter that determines the step size during weight updates."
    },
    {
      "question": "SA Q41: What is 'gradient clipping' and why is it used?",
      "answer": "Gradient clipping limits the magnitude of gradients to prevent exploding gradients during training."
    },
    {
      "question": "SA Q42: What does 'parameter sharing' mean in CNNs?",
      "answer": "Parameter sharing means using the same filter (set of weights) across different regions of the input."
    },
    {
      "question": "SA Q43: What is a 'residual connection' in deep neural networks?",
      "answer": "A residual connection is a shortcut that adds the input of a layer to its output to help gradient flow."
    },
    {
      "question": "SA Q44: What is the purpose of the softmax function in classification?",
      "answer": "Softmax converts raw scores into a probability distribution over classes."
    },
    {
      "question": "SA Q45: What is meant by 'minibatch gradient descent'?",
      "answer": "Minibatch gradient descent updates model weights using a small subset of the training data at each step."
    },
    {
      "question": "SA Q46: What does 'data augmentation' refer to?",
      "answer": "Data augmentation is the process of creating new training samples by applying transformations to existing data."
    },
    {
      "question": "SA Q47: Define 'feature map' in CNNs.",
      "answer": "A feature map is the output of a convolutional layer that represents detected features in the input."
    },
    {
      "question": "SA Q48: What is the role of a pooling layer in a CNN?",
      "answer": "A pooling layer reduces the spatial dimensions of feature maps while retaining important information."
    },
    {
      "question": "SA Q49: What does 'backpropagation through time' (BPTT) refer to?",
      "answer": "BPTT is the process of applying backpropagation to a recurrent neural network by unrolling it over time."
    },
    {
      "question": "SA Q50: What is a 'discriminator' in a GAN?",
      "answer": "The discriminator is the network that learns to differentiate between real and generated data."
    },
    {
      "question": "SA Q51: What does 'gradient descent' optimize?",
      "answer": "Gradient descent minimizes the loss function by iteratively updating the model's parameters."
    },
    {
      "question": "SA Q52: What is 'AdaGrad'?",
      "answer": "AdaGrad is an optimizer that adapts the learning rate for each parameter based on past gradients."
    },
    {
      "question": "SA Q53: What is 'RMSProp'?",
      "answer": "RMSProp is an optimizer that adjusts learning rates by dividing by a running average of recent gradients' magnitudes."
    },
    {
      "question": "SA Q54: What does 'weight decay' refer to?",
      "answer": "Weight decay is a regularization technique that adds a penalty proportional to the magnitude of the weights."
    },
    {
      "question": "SA Q55: What is the purpose of using an activation function in a neural network?",
      "answer": "An activation function introduces non-linearity, enabling the network to learn complex patterns."
    },
    {
      "question": "SA Q56: What does 'stride' refer to in convolution operations?",
      "answer": "Stride is the step size with which the convolution filter moves over the input."
    },
    {
      "question": "SA Q57: What is 'padding' in the context of convolutional layers?",
      "answer": "Padding involves adding extra pixels (often zeros) around the input to control the spatial size of the output."
    },
    {
      "question": "SA Q58: What does 'inference' mean in deep learning?",
      "answer": "Inference is the process of using a trained model to make predictions on new data."
    },
    {
      "question": "SA Q59: What is meant by 'model generalization'?",
      "answer": "Generalization is the ability of a model to perform well on unseen data."
    },
    {
      "question": "SA Q60: Define 'hyperparameter' in machine learning.",
      "answer": "A hyperparameter is a configuration setting used to control the learning process, such as learning rate or batch size."
    },
    {
      "question": "SA Q61: What does 'fine-tuning' involve in transfer learning?",
      "answer": "Fine-tuning involves updating the weights of a pre-trained model on a new task with a smaller learning rate."
    },
    {
      "question": "SA Q62: What is 'dropout rate'?",
      "answer": "Dropout rate is the probability that a neuron is temporarily deactivated during training."
    },
    {
      "question": "SA Q63: What does 'epoch' mean in training neural networks?",
      "answer": "An epoch is one complete pass through the entire training dataset."
    },
    {
      "question": "SA Q64: What is 'early stopping' and why is it used?",
      "answer": "Early stopping is a method to halt training when performance on a validation set starts to worsen, preventing overfitting."
    },
    {
      "question": "SA Q65: What is the main goal of reinforcement learning?",
      "answer": "The main goal is to learn a policy that maximizes cumulative reward over time."
    },
    {
      "question": "SA Q66: What is 'Q-learning'?",
      "answer": "Q-learning is a reinforcement learning algorithm that learns the value of action-state pairs to guide decision making."
    },
    {
      "question": "SA Q67: What does 'policy gradient' refer to?",
      "answer": "Policy gradient methods optimize the policy directly by computing gradients of the expected reward."
    },
    {
      "question": "SA Q68: What is meant by 'experience replay' in deep reinforcement learning?",
      "answer": "Experience replay stores past experiences and reuses them to break correlation during training."
    },
    {
      "question": "SA Q69: What is 'knowledge distillation'?",
      "answer": "Knowledge distillation transfers knowledge from a large, complex model to a smaller, more efficient one."
    },
    {
      "question": "SA Q70: Define 'ensemble learning'.",
      "answer": "Ensemble learning combines predictions from multiple models to improve overall performance."
    },
    {
      "question": "SA Q71: What does 'latent space' refer to in autoencoders?",
      "answer": "Latent space is the compressed, lower-dimensional representation learned by the encoder."
    },
    {
      "question": "SA Q72: What is 'sparsity' in the context of neural networks?",
      "answer": "Sparsity refers to a situation where most of the weights or activations are zero or near zero."
    },
    {
      "question": "SA Q73: What does 'convergence' mean during training?",
      "answer": "Convergence means that the loss has stabilized and further training does not significantly improve performance."
    },
    {
      "question": "SA Q74: What is the purpose of using a validation set?",
      "answer": "A validation set is used to tune hyperparameters and monitor model performance to prevent overfitting."
    },
    {
      "question": "SA Q75: What does 'epoch' refer to?",
      "answer": "An epoch is one complete cycle through the entire training dataset."
    },
    {
      "question": "SA Q76: What is 'mini-batch gradient descent'?",
      "answer": "Mini-batch gradient descent updates the model weights using a small subset of the training data at each iteration."
    },
    {
      "question": "SA Q77: Define 'learning rate'.",
      "answer": "The learning rate is a hyperparameter that determines the size of the weight updates during training."
    },
    {
      "question": "SA Q78: What is 'model capacity'?",
      "answer": "Model capacity is the ability of a model to fit a wide variety of functions."
    },
    {
      "question": "SA Q79: What is 'overparameterization' in neural networks?",
      "answer": "Overparameterization means having more parameters than necessary, which can lead to overfitting."
    },
    {
      "question": "SA Q80: What does 'stochastic' mean in stochastic gradient descent?",
      "answer": "It means that weight updates are based on randomly selected subsets of data."
    },
    {
      "question": "SA Q81: What is the purpose of using a softmax activation in the output layer?",
      "answer": "Softmax converts logits into a probability distribution over classes."
    },
    {
      "question": "SA Q82: Define 'activation function'.",
      "answer": "An activation function introduces non-linearity into a neural network."
    },
    {
      "question": "SA Q83: What is 'backpropagation through time'?",
      "answer": "It is the extension of backpropagation for training recurrent neural networks by unrolling them in time."
    },
    {
      "question": "SA Q84: What does 'regularization' aim to achieve in model training?",
      "answer": "Regularization aims to reduce overfitting by penalizing model complexity."
    },
    {
      "question": "SA Q85: What is the role of the 'optimizer' in deep learning?",
      "answer": "The optimizer adjusts model weights to minimize the loss function."
    },
    {
      "question": "SA Q86: What does 'parameter tuning' refer to?",
      "answer": "Parameter tuning involves adjusting hyperparameters to improve model performance."
    },
    {
      "question": "SA Q87: What is a 'hyperparameter'?",
      "answer": "A hyperparameter is a configuration variable set before training that governs the learning process."
    },
    {
      "question": "SA Q88: What is 'data augmentation'?",
      "answer": "Data augmentation is the process of creating additional training samples through transformations."
    },
    {
      "question": "SA Q89: What is the main function of an embedding layer in NLP?",
      "answer": "An embedding layer converts discrete tokens into dense vector representations."
    },
    {
      "question": "SA Q90: What does 'model generalization' refer to?",
      "answer": "Generalization is the model's ability to perform well on unseen data."
    },
    {
      "question": "SA Q91: What is a 'confusion matrix'?",
      "answer": "A confusion matrix is a table used to evaluate the performance of a classification model."
    },
    {
      "question": "SA Q92: What does 'precision' measure in classification tasks?",
      "answer": "Precision measures the proportion of correct positive predictions among all positive predictions."
    },
    {
      "question": "SA Q93: What does 'recall' measure in classification tasks?",
      "answer": "Recall measures the proportion of actual positives correctly identified by the model."
    },
    {
      "question": "SA Q94: Define 'F1-Score'.",
      "answer": "The F1-Score is the harmonic mean of precision and recall."
    },
    {
      "question": "SA Q95: What is 'gradient clipping' used for?",
      "answer": "Gradient clipping is used to prevent exploding gradients during training."
    },
    {
      "question": "SA Q96: What does 'weight initialization' affect in neural networks?",
      "answer": "It affects the starting point of optimization and can influence convergence speed and performance."
    },
    {
      "question": "SA Q97: What is the purpose of a 'target network' in deep reinforcement learning?",
      "answer": "A target network provides stable targets by being updated less frequently than the main network."
    },
    {
      "question": "SA Q98: What is 'knowledge distillation' in deep learning?",
      "answer": "Knowledge distillation transfers knowledge from a large model to a smaller model."
    },
    {
      "question": "SA Q99: What does 'ensemble learning' mean?",
      "answer": "Ensemble learning combines multiple models to improve prediction performance."
    },
    {
      "question": "SA Q100: What is the primary benefit of using GPUs in deep learning?",
      "answer": "GPUs accelerate computation by parallelizing operations, which speeds up training."
    }
  ]
};

  
  // If running in Node.js environment, export the database
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = questionsDatabase;
  }