# e3nn-Born-prediction


## Introduction
An equivariant graph neural network that are trainined to predict the Born effective charge matrices and the dielectric constant tensors of a crystal.
These two quantities contribute to the long-range dipole-dipole interactions that correct the vibrational phonon spectrum near $\Gamma$ point with $\vec{q} \rightarrow 0$, which leads to the non-analytic corrections (NAC) / LO-TO splittings.

$$D_{\alpha\beta}(jj',\mathbf{q}\to \mathbf{0}) = D_{\alpha\beta}(jj',\mathbf{q}=\mathbf{0}) + \frac{1}{\sqrt{m_j m_{j'}}} \frac{4\pi}{\Omega_0} \frac{ \left[  \sum_{\gamma} q_{\gamma} Z_{j,\gamma\alpha} \right] \left[ \sum_{\gamma} q_{\gamma} Z_{j',\gamma' \beta} \right] } {\sum_{ \alpha \beta } q_{ \alpha } \epsilon_{ \alpha \beta }^{ \infty } q_{ \beta }}$$

with Born effective charges $Z$ and dielectric constants $\epsilon^{ \infty }$.
 




## Dataset
The training data is built from the phonon database, calculated with density functional perturbation theory (DFPT) approach, to get the linear responses.



## Installation




## Usage

Here we have provided the trained model weights.



## References




