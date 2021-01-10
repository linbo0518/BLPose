# BLPose (BaseLine Pose Estimation)

![love](https://img.shields.io/badge/💖-build%20with%20love-blue.svg?style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/linbo0518/BLPose?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/linbo0518/BLPose?style=for-the-badge)

PyTorch's Pose Estimation Toolbox

## Requirement

- Python 3
- PyTorch >= 1.0.0

<!-- ## Quick Start

Execute the following command in your terminal

```sh
pip install --upgrade git+https://github.com/linbo0518/BLPose.git
```

## Documentation

For more information, please see [Documentation](Documentation.md) -->

## Supported Module

- Backbone
  - VGG
    - VGG11
    - VGG13
    - VGG16
    - VGG19
  - ResNet
    - ResNet18
    - ResNet34
    - ResNet50
    - ResNet101
    - ResNet152
  - SE ResNet
    - SE ResNet50
    - SE ResNet101
    - SE ResNet152
  - MobileNet v1 (1.0)
  - MobileNet v2 (1.0)
- Model
  - OpenPose
- Metric
  - Average Meter
- Others
  - Xavier/MSRA initialization (support zero gamma in last BatchNorm)
  - Mixed precision training
  - Online Hard Example Mining
  - Precise BatchNorm (comming soon...)

| Backbone \ Model |

<!-- ## Analysis

- Parameters

| Backbone \ Model |

- Multiply-accumulate operations (MACs)

| Backbone \ Model | -->

## Changelog

See [Changelog](Changelog.md)

## References

- Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv
  preprint arXiv:1409.1556 (2014).
- Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv
  preprint arXiv:1704.04861 (2017).
- Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE Conference on
  Computer Vision and Pattern Recognition. 2018.
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer
  vision and pattern recognition. 2016.
- Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer
  vision and pattern recognition. 2018.
