#!/usr/bin/env python
# coding: utf-8

# In[88]:


# Support for simulating a vector field over a hypercube, with a known value at
# a selected point.
#
# Carlo Graziani, ANL
#
"""
Support for simulating a vector field over a hypercube, with a known value at a
selected point.
"""

import torch
import numpy as np
from scipy.stats import uniform
import math


######################################################################
######################################################################
######################################################################
class VField(object):
    """
    An N-dimensional vector field over a D-dimensional hypercube, with
    a given value at a given point.
    """
#########################################
    def __init__(self, N=2, D=2, tgt_loc=np.array([0.2,0.1]),
                 tgt_vec=np.array([0.5,1.0]),
                 polynomial_order=4, polynomial_coefficients=None):

        """
        Constructor

        Args:

        N (int): Dimension of vector field. Default: 2

        D (int): Dimension domain hypercube. Default: 2

        tgt_loc (ndarray(D)): Location of target value. Default:
        np.array([0.2,0.1])

        tgt_vec (ndarray(N)): Target value. Default: np.array([0.5,1.0])

        polynomial_order (int): Order of random polynomial chosen for vector
        field component functions. Default: 4

        polynomial_coefficients (None or dict): If
        None, the polynomial coefficients are chosen randomly and the resulting
        polynomial is adjusted to give the target value at the target location.
        The resulting coefficients are stored in a dict at self.p_coef.
        Otherwise, the polynomial from the provided coefficients is adjusted to
        give the target value at the target location, coded in a dict as
        expected by the code. Default: None
        """

        self.N = N
        self.D = D
        self.tgt_loc = tgt_loc
        self.tgt_vec = tgt_vec
        self.polynomial_order = polynomial_order

        if polynomial_coefficients is None:
            self._set_pcoef()
        else:
            self.pcoef = polynomial_coefficients
            
        # Adjust coefficients to hit target

        v0 = self._vf(tgt_loc)

        self.pcoef[0][:] = self.pcoef[0][:] - v0 + tgt_vec

#########################################
    def _set_pcoef(self):
        """
        Set polynomial coefficients randomly
        """

        self.pcoef={}
        self.pcoef[0] = uniform.rvs(size=self.N)

        for m in range(1,self.polynomial_order+1):
            shp = np.empty(m+1, dtype=int)
            shp[0] = self.N
            shp[1:] = self.D
            sz=self.D**m * self.N
            arr = uniform.rvs(size=sz).reshape(shp)
            self.pcoef[m] = arr


#########################################
    def _vf(self, loc):
        """
        Compute the value of the vector field at location loc
        """

        v = self.pcoef[0]
        mf = 1
        for m in range(1,self.polynomial_order+1):
            mf *= m
            c = np.copy(self.pcoef[m])
            for l in range(m):
                c = np.dot(c, loc)
            v = v + c / mf

        return v

#########################################
    def __call__(self, loc):
        """
        Return the value of the vector field at location loc
        """
        x = loc
        x = x.reshape(x.shape[0], self.D)
        out = torch.zeros(x.shape[0], self.N)
        for i in range(x.shape[0]):
            out[i] = torch.Tensor(self._vf(x[i])) + torch.randn(torch.Tensor(self._vf(x[i])).size()) * math.sqrt(0.04)
        return out


# In[90]:


x = np.random.rand(10000,2)
vfield = VField()
y = vfield(x)
y.shape
print(y)


# In[91]:


y.shape


# In[ ]:




