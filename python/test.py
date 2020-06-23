"""Script to test gpuNUFFT wrapper.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
from gpuNUFFT import NUFFTOp
import time


def get_nufft_op(kspace_loc, img_size, n_coils, n_interpolators):
    sens_maps=None
    return NUFFTOp(
        np.reshape(kspace_loc, kspace_loc.shape[::-1], order='F'),
        img_size,
        n_coils,
        sens_maps,
        weights,
        3,
        8,
        2,
        True,
    )


def setUp():
    """ Setup variables:
    N = Image size
    max_iter = Number of iterations to test
    num_channels = Number of channels to be tested with for
                    multichannel tests
    """
    # IMAGE
    np.random.seed(0)
    img_size = [384, 384, 208]
    [x, y, z] = np.meshgrid(np.linspace(-1, 1, img_size[0]),
                            np.linspace(-1, 1, img_size[1]),
                            np.linspace(-1, 1, img_size[2]))
    img = (x**2 + y**2 + z**2 < 0.5**2)
    img = img.astype(np.complex64)

    # KCOORDS
    kspace_loc = np.random.random((4096*4096, 3))
    num_interpolators = 21
    # WEIGHTS
    weights = np.ones((num_interpolators, kspace_loc.shape[0]))
    print('Input weights shape is', weights.shape)

    # COIL MAPS
    n_coils = 42
    coil_maps = ((1 / (x**2 + y**2 + z**2 + 1))).astype(np.complex64)
    smaps = np.tile(coil_maps, (n_coils, 1, 1, 1))
    multi_img = np.tile(img, (n_coils, 1, 1, 1))
    return kspace_loc, multi_img, weights

print('Apply forward op')
kspace_loc, multi_img, weights = setUp()
operator = get_nufft_op(kspace_loc, multi_img.shape[1:], multi_img.shape[0], weights.shape[0])
x = operator.op(np.asarray(
    [np.reshape(image_ch.T, image_ch.size) for image_ch in multi_img]
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
