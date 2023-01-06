# Comparative Study of Classical Medical Image Segmentation Algorithms
# Foobar

This file contains the implementation of the project assignment for class CS554 (Computer Vision), offered in Fall 2022, at Bilkent University. This assignment covers the implementation of the following four tasks:

- Conditional Random Field (Codes & Results in the form of a Jupyter File)
- GrabCut (Codes & Results in the form of a Jupyter File)
- K-means Clustering (Codes in Matlab File & Results in a zip form.)
- Mean-shift Clustering (Codes in Matlab File & Results in a zip form.)

## Conditional Random Field
Conditional random fields (CRFs) are undirected probabilistic graphical models. They are, by definition, discriminative. We have implemented a similar version presented in "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials." However, to reduce the computation cost, instead of using DenseCRF, we utilized the 8-neighborhood model.

## GrabCut
The GrabCut algorithm was designed by Carsten Rother, Vladimir Kolmogorov & Andrew Blake from Microsoft Research Cambridge, UK. The whole pipeline has been implemented in Python. However, this is not a full implementation, as the original paper has additional features for fine-tuning the segmentation.

## K-means Clustering
In K-mean segmentation, the number of clusters is given
as input with the image. In our algorithm, we determine initial
cluster means by using two different initializations, uniform
and random. Then, using the L1 norm, determine the new
cluster mean through some iterations.

## Mean-shift Clustering 
In Mean Shift clustering, the number of clusters is not given
as input. Each pixel in the image is assigned to the nearer
peak (mode) point. The peak or mode point is determined
based on the density map of the feature space of the image. The
number of peak points determines the number of clusters.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following libraries.

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install os
pip install scipy
pip install sklearn.cluster
pip install igraph
pip install sklearn.metrics
pip install typing
pip install Pillow
```

## Contributing

The contributors to this implementation are as follows:

- Atakan Topcu ([linkedin](https://www.linkedin.com/in/atakan-topcu-0a47791b9/))
- Ege Kor ([linkedin](https://www.linkedin.com/in/ege-kor-691b5b192/))
