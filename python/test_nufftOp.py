"""Script to test gpuNUFFT wrapper.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from gpuNUFFT import NUFFTOp
import unittest
import time


class TestgpuNUFFT(unittest.TestCase):
    """ Test the adjoint operator of the Wavelets both for 2D and 3D.
    """

    def get_nufft_op(self, sens_maps=None):
        return NUFFTOp(
            np.reshape(self.kspace_loc, self.kspace_loc.shape[::-1], order='F'),
            self.img_size,
            self.n_coils,
            sens_maps,
            self.weights,
            3,
            8,
            2,
            True,
        )

    def setUp(self):
        """ Setup variables:
        N = Image size
        max_iter = Number of iterations to test
        num_channels = Number of channels to be tested with for
                        multichannel tests
        """
        # IMAGE
        np.random.seed(0)
        self.img_size = [384, 384, 208]
        [x, y, z] = np.meshgrid(np.linspace(-1, 1, self.img_size[0]),
                                np.linspace(-1, 1, self.img_size[1]),
                                np.linspace(-1, 1, self.img_size[2]))
        img = (x**2 + y**2 + z**2 < 0.5**2)
        self.img = img.astype(np.complex64)

        # KCOORDS
        self.kspace_loc = np.random.random((4096*4096, 3))

        # WEIGHTS
        self.weights = np.ones(self.kspace_loc.shape[0])
        print('Input weights shape is', self.weights.shape)

        # COIL MAPS
        self.n_coils = 42
        coil_maps = ((1 / (x**2 + y**2 + z**2 + 1))).astype(np.complex64)
        self.smaps = np.tile(coil_maps, (self.n_coils, 1, 1, 1))
        self.multi_img = np.tile(img, (self.n_coils, 1, 1, 1))

    def test_multicoil_with_sense(self):
        print('Apply forward op')
        operator = self.get_nufft_op(self.smaps)
        x = operator.op(np.reshape(self.img.T, self.img.size))
        y = np.random.random(x.shape)
        print('Output kdata shape is', x.shape)
        print('-------------------------------')
        print('Apply adjoint op')
        st = time.time()
        img_adj = operator.adj_op(x)
        print(time.time() - st)
        adj_y = operator.adj_op(y)
        print('Output adjoint img shape is', img_adj.shape)
        img_adj = np.squeeze(img_adj).T
        adj_y = np.squeeze(adj_y).T
        print(img_adj.shape)
        plt.figure(3)
        plt.imshow(abs(img_adj))
        plt.title('adjoint image')
        plt.show()
        # Test Adjoint property
        x_d = np.vdot(self.img, adj_y)
        x_ad = np.vdot(x, y)
        np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
        print('done')

    def test_multicoil_without_sense(self):
        print('Apply forward op')
        operator = self.get_nufft_op()
        x = operator.op(np.asarray(
            [np.reshape(image_ch.T, image_ch.size) for image_ch in self.multi_img]
        ).T)
        y = np.random.random(x.shape)
        print('Output kdata shape is', x.shape)
        print('-------------------------------')
        print('Apply adjoint op')
        st = time.time()
        img_adj = operator.adj_op(x)
        print(time.time() - st)
        print('Output adjoint img shape is', img_adj.shape)
        img_adj = np.squeeze(img_adj)
        img_adj = np.asarray(
                [image_ch.T for image_ch in img_adj]
            )
        adj_y = np.squeeze(operator.adj_op(y))
        adj_y = np.asarray(
                [image_ch.T for image_ch in adj_y]
            )
        print(img_adj.shape)
        plt.figure(3)
        plt.imshow(abs(img_adj[1]))
        plt.title('adjoint image')
        plt.show()
        # Test Adjoint property
        x_d = np.vdot(self.multi_img, adj_y)
        x_ad = np.vdot(x, y)
        np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
        print('done')
