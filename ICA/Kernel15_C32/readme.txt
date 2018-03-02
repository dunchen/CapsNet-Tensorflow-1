1. run FashionUnsu.py to generate receptive fields (local conv weights)
this is stored in pretrainW_ksize_15_stride_1_channel_32.npy
(file name may vary with hyperparameters)

2. run lc_mnist_pre_lasttwo.py for training without modifying local conv weights

3. run cnn_mnist.py to run a comparable CNN classifier

Fashion MNIST data are stored in ./tmp/data
*.gz files are for FashionUnsu.py pretraining