# Graph-Generative-Flow

Code for reproducing results in 'Graph-based Normalizing Flow for Human Motion
Generation and Reconstruction'

[[paper](https://ui.adsabs.harvard.edu/abs/2021arXiv210403020Y/abstract)] [[data](https://github.com/simonalexanderson/StyleGestures)]

If this code helps with your work, please cite:

```bibtex
@article{yin2021graph,
  title={Graph-based Normalizing Flow for Human Motion Generation and Reconstruction},
  author={Yin, Wenjie and Yin, Hang and Kragic, Danica and Bj{\"o}rkman, M{\aa}rten},
  journal={arXiv preprint arXiv:2104.03020},
  year={2021}
}
```

## Dataset

The data is pooled from the Edinburgh Locomotion, CMU Motion Capture, and HDM05 datasets.
Thanks for Gustav Eje Henter, Simon Alexanderson, and Jonas Beskow originally sharing the data [here](https://github.com/simonalexanderson/StyleGestures).


## Methods

We use a spatial convolutional network in the affine coupling layer to extract skeleton features. The conditioning information include the past poses. All these are concatenated as one vector in MoGlow. We use a spatial temporal convolution networks to extract the features of past sequence. To reconstruct the missing data, we first generate future poses, then reverse the generated poses and control signals. Regarding the reversed sequences as control information to generate markers to fill the holes of the missing data. 

![network](https://github.com/YIN95/Graph-Generative-Flow/blob/main/media/image17.png)


