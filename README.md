
## Improving Sound Event Classification by <br> Increasing Shift Invariance in Convolutional Neural Networks

This repository contains the TensorFlow implementation of the models and pooling mechanisms proposed in the following <a href="https://arxiv.org/abs/2107.00623" target="_blank">paper</a>. If you use this code or part of it, please cite:

>Eduardo Fonseca, Andres Ferraro, Xavier Serra, "Improving Sound Event Classification by Increasing Shift Invariance in Convolutional Neural Networks", arXiv:2107.00623, 2021.


Specifically, the implementations included are

- models: VGG41 and VGG42 (see `model.py`)
- pooling mechanisms to increase shift invariance in convolutional neural networks:
  - Trainable Low-Pass Filters (TLPF) and BlurPool, inspired by [1], (see `lpf.py`)
  - Adaptive Polyphase Sampling (APS), inspired by [2], (see `aps.py`)


### Citation
```
@article{fonseca2021shift,
  title={Improving Sound Event Classification by Increasing Shift Invariance in Convolutional Neural Networks},
  author={Fonseca, Eduardo and Ferraro, Andres and Serra, Xavier},
  journal={arXiv preprint arXiv:2107.00623},
  year={2021}
}
```
### Questions?
Please contact Edu Fonseca with comments and questions `efonseca at google dot com`.


### References

[1] R. Zhang, “Making convolutional networks shift-invariant again,” in International Conference on Machine Learning. PMLR, 2019, pp.
7324–7334.

[2] A. Chaman and I. Dokmanic, “Truly shift-invariant convolutional neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3773–3783.