# NYU-Deep-Learning-Mini-Project-S24
Advanced Implementation of ResNet-18 for CIFAR-10 classification 

## Requirements
- Python 3.6+
- PyTorch 1.6.0+

## Usage
1. Train

```
python main.py
```

2. Test, visualization and verification

When your training is done, You can run the Jupyter notebook file `project.ipynb` with clear visualization plots and results.

## Note
If you want to specify GPU to use, you should set environment variable `CUDA_VISIBLE_DEVICES=0`, for example.

## References
- Krizhevsky, A., Hinton, G., & others. (2009). Learning multiple layers of features from tiny images. Toronto, ON, Canada.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770–778).
- DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.
- Loshchilov, I., & Hutter, F. (2016). Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.
- Sanghyun, W., Jongchan, P., & others  (2018). CBAM: Convolutional Block Attention Module. arXiv preprint arXiv:1807.06521v2