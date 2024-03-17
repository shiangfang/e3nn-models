# e3nn-models

Here we use e3nn to build the following generalized models based on the NequIP architecture:

1. Generalized energy model to include the electric field dependence for molecules.

2. Tensorial models to predict the Born effective charges (node prediction task) and the dielectric matrices (graph prediction task) relevant for the electric dipole-dipole interactions that affect the phonons near $\vec{q} \rightarrow 0$.




## Quick Start

The following steps will guid you through the installation and the tutorial notebooks.

1. In a conda env with JAX / CUDA available clone and download this repository.

2. In the downloaded directory run:
    ```bash
    pip install -r requirements.txt
    ```

## Introduction and Tutorials
### Generalized energy model to include electric field dependence for molecules
By generalizing the energy functional to include additional control variables such as the external electric field, one can relate the energy functional with various physical quantities such as the electric polarization and the polarizability via the auto-differentiation with respect to these additional variables.
In this model, we demonstrate this with an ethanol molecule dataset generated with electric field that modifies the energy and atomic forces.

For the tutorial, please check out the notebook e3nn-Efield-potential.ipynb

### Tensorial models to predict the Born effective charges and the dielectric matrices 
An equivariant graph neural network that are trainined to predict the Born effective charge matrices and the dielectric constant tensors of a crystal.
These two quantities contribute to the long-range dipole-dipole interactions that correct the vibrational phonon spectrum near $\Gamma$ point with $\vec{q} \rightarrow 0$, which leads to the non-analytic corrections (NAC) / LO-TO splittings.

$$D_{\alpha\beta}(jj',\mathbf{q}\to \mathbf{0}) = D_{\alpha\beta}(jj',\mathbf{q}=\mathbf{0}) + \frac{1}{\sqrt{m_j m_{j'}}} \frac{4\pi}{\Omega_0} \frac{ \left[  \sum_{\gamma} q_{\gamma} Z_{j,\gamma\alpha} \right] \left[ \sum_{\gamma} q_{\gamma} Z_{j',\gamma' \beta} \right] } {\sum_{ \alpha \beta } q_{ \alpha } \epsilon_{ \alpha \beta }^{ \infty } q_{ \beta }}$$ with Born effective charges $Z$ and dielectric constants $\epsilon^{ \infty }$.
While the Born effective charge model is a nodal prediction task, the dielectric constant matrix modeling is a graph level prediction task.
The models here are constructed based on the NequIP architecture to generate tensorial output by retaining latent state vectors with higher angular momentum L.

For the tutorials, please check out the e3nn-dipdip-eps.ipynb and e3nn-dipdip-becs.ipynb for training the models to predict the dielectric constant matrix and Born effective charges respectively.


## References

1. [e3nn: Euclidean Neural Networks](https://arxiv.org/abs/2207.09453)

2. [NequIP: E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials](https://www.nature.com/articles/s41467-022-29939-5)

3. [JAX M.D. A Framework for Differentiable Physics](https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf)

## Citation

1. [NeurIPS 2023 AI4Mat workshop](https://openreview.net/forum?id=xxyHjer00Y)







