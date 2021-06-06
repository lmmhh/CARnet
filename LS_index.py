#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import division, absolute_import, print_function

import numpy as np
import cv2
import math

from scipy.ndimage import uniform_filter, gaussian_filter
import warnings

# After numpy 1.15, a new backward compatible function have been implemented.
# See https://github.com/numpy/numpy/pull/11966
from distutils.version import LooseVersion as Version
old_numpy = Version(np.__version__) < Version('1.16')
if old_numpy:
    from numpy.lib.arraypad import _validate_lengths
else:
    from numpy.lib.arraypad import _as_pairs

__all__ = ['LSIndex']


_integer_types = (np.byte, np.ubyte,          # 8 bits
                  np.short, np.ushort,        # 16 bits
                  np.intc, np.uintc,          # 16 or 32 or 64 bits
                  np.int_, np.uint,           # 32 or 64 bits
                  np.longlong, np.ulonglong)  # 64 bits
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
dtype_range.update(_integer_ranges)


def warn(message, category=None, stacklevel=2):
    """A version of `warnings.warn` with a default stacklevel of 2.
    """
    if category is not None:
        warnings.warn(message, category=category, stacklevel=stacklevel)
    else:
        warnings.warn(message, stacklevel=stacklevel)


def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),)`` specifies a fixed start and end crop
        for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.

    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)
    if old_numpy:
        crops = _validate_lengths(ar, crop_width)
    else:
        crops = _as_pairs(crop_width, ar.ndim, as_index=True)
    slices = tuple(slice(a, ar.shape[i] - b)
                   for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


def gradient(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=-1)
    value, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return value, angle


class LSIndex(object):
    """Compute the LS index between two images.

        Parameters
        ----------
        X, Y : ndarray
            Image.  Any dimensionality. X: original image Y: compressed image
        win_size : int or None
            The side-length of the sliding window used in comparison.  Must be an
            odd value.  If `gaussian_weights` is True, this is ignored and the
            window size will depend on `sigma`.
        data_range : float, optional
            The data range of the input image (distance between minimum and
            maximum possible values).  By default, this is estimated from the image
            data-type.
        multichannel : bool, optional
            If True, treat the last dimension of the array as channels. Similarity
            calculations are done independently for each channel then averaged.
        gaussian_weights : bool, optional
            If True, each patch has its mean and variance spatially weighted by a
            normalized Gaussian kernel of width sigma=1.5.
        full : bool, optional
            If True, also return the full structural similarity image.

        Other Parameters
        ----------------
        use_sample_covariance : bool
            if True, normalize covariances by N-1 rather than, N where N is the
            number of pixels within the sliding window.
        K1 : float
            algorithm parameter, K1 (small constant, see [1]_)
        K2 : float
            algorithm parameter, K2 (small constant, see [1]_)
        r : float
            algorithm parameter, R (constant [0,1], rate)
        sigma : float
            sigma for the Gaussian when `gaussian_weights` is True.

        Notes
        -----
        To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
        to True, `sigma` to 1.5, and `use_sample_covariance` to False.

        References
        ----------
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           DOI:10.1109/TIP.2003.819861

        .. [2] Avanaki, A. N. (2009). Exact global histogram specification
           optimized for structural similarity. Optical Review, 16, 613-621.
           http://arxiv.org/abs/0901.0065,
           DOI:10.1007/s10043-009-0119-z
        Returns
        -------
        mssim : float
            The mean structural similarity over the image.
        psnr : float
            The psnr over the image.
        S : ndarray
            The full SSIM image.  This is only returned if `full` is set to True.
        """
    def __init__(self, win_size=None, multichannel=False, gaussian_weights=False, **kwargs):
        self.win_size = win_size
        # self.data_range = data_range
        self.multichannel = multichannel
        # self.gaussian_weights = gaussian_weights
        sigma = kwargs.pop('sigma', 1.5)
        if sigma < 0:
            raise ValueError("sigma must be positive")

        if win_size is None:
            if gaussian_weights:
                win_size = 11  # 11 to match Wang et. al. 2004
            else:
                win_size = 7  # backwards compatibility

        if not (win_size % 2 == 1):
            raise ValueError('Window size must be odd.')

        if gaussian_weights:
            # sigma = 1.5 to approximately match filter in Wang et. al. 2004
            # this ends up giving a 13-tap rather than 11-tap Gaussian
            self.filter_func = gaussian_filter
            self.filter_args = {'sigma': sigma}

        else:
            self.filter_func = uniform_filter
            self.filter_args = {'size': win_size}

    def check(self, X, Y, data_range):
        if not X.shape == Y.shape:
            raise ValueError('Input images must have the same dimensions.')
        if np.any((np.asarray(X.shape) - self.win_size) < 0):
            raise ValueError(
                "win_size exceeds image extent.  If the input is a multichannel "
                "(color) image, set multichannel=True.")
        if data_range is None:
            if X.dtype != Y.dtype:
                warn("Inputs have mismatched dtype.  Setting data_range based on "
                     "im_true.")
            dmin, dmax = dtype_range[X.dtype.type]
            true_min, true_max = np.min(X), np.max(X)
            if true_max > dmax or true_min < dmin:
                raise ValueError(
                    "im_true has intensity values outside the range expected for "
                    "its data type.  Please manually specify the data_range")
            if true_min >= 0:
                # most common case (255 for uint8, 1 for float)
                data_range = dmax
            else:
                data_range = dmax - dmin
        return data_range

    def psnr(self, X, Y, data_range=None, **kwargs):

        fac = kwargs.pop('factor', 1e-10)
        data_range = self.check(X, Y, data_range)
        mse = np.mean(np.square(X - Y), dtype=np.float64)
        psnr1 = 10 * np.log10((data_range ** 2) / mse)

        GXx = np.abs(cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=-1))
        GXy = np.abs(cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=-1))
        GYx = np.abs(cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=-1))
        GYy = np.abs(cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=-1))

        uGXx = self.filter_func(GXx, **self.filter_args)
        uGXy = self.filter_func(GXy, **self.filter_args)
        uGYx = self.filter_func(GYx, **self.filter_args)
        uGYy = self.filter_func(GYy, **self.filter_args)

        dx = uGXy - uGXx
        dy = uGYy - uGYx
        dd = dy - dx
        R = math.ceil(max(np.max(dx), np.max(dy)) - min(np.min(dx), np.min(dy)))
        if R < 0:
            raise ValueError('R must >=0.')
        # dd[dd < 0] = 0  # fac * R
        mse2 = np.mean(np.square(dd))
        psnr2 = 10 * np.log10((R ** 2) / mse2)
        r = kwargs.pop('rate', 0.3)
        if r < 0 or r > 1:
            raise ValueError("r must between [0,1]")
        psnr = (1 - r) * psnr1 + r * psnr2
        #return mse2, R
        return psnr

    def ssim(self, X, Y, data_range=None, full=False, **kwargs):
        data_range = self.check(X, Y, data_range)
        if self.multichannel:
            # loop over channels
            args = dict(data_range=data_range,
                        multichannel=False,
                        full=full)
            args.update(kwargs)
            nch = X.shape[-1]
            mssim = np.empty(nch)
            if full:
                S = np.empty(X.shape)
            for ch in range(nch):
                ch_result = self.ssim(X[..., ch], Y[..., ch], **args)
                if full:
                    mssim[..., ch], S[..., ch] = ch_result
                else:
                    mssim[..., ch] = ch_result
            mssim = mssim.mean()
            if full:
                return mssim, S
            else:
                return mssim
        use_sample_covariance = kwargs.pop('use_sample_covariance', True)
        K1 = kwargs.pop('K1', 0.01)
        K2 = kwargs.pop('K2', 0.03)
        r = kwargs.pop('rate', 0.5)
        if K1 < 0:
            raise ValueError("K1 must be positive")
        if K2 < 0:
            raise ValueError("K2 must be positive")
        if r < 0 or r > 1:
            raise ValueError("r must between [0,1]")
        ndim = X.ndim
        NP = self.win_size ** ndim
        # filter has already normalized by NP
        if use_sample_covariance:
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        _, Xangle = gradient(X)
        _, Yangle = gradient(Y)
        S1, mssim1 = self.ssim_(X, Y, cov_norm, data_range, K1, K2)
        S2, mssim2 = self.ssim_(Xangle, Yangle, cov_norm, 360, K1, K2)
        S = (1 - r) * S1 + r * S2
        mssim = (1 - r) * mssim1 + r * mssim2
        if full:
            return mssim, S
        else:
            return mssim

    def ssim_(self, X, Y, cov_norm, data_range, K1, K2):
        # ndimage filters need floating point data
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        # compute (weighted) means
        ux = self.filter_func(X, **self.filter_args)
        uy = self.filter_func(Y, **self.filter_args)
        uxx = self.filter_func(X * X, **self.filter_args)
        uyy = self.filter_func(Y * Y, **self.filter_args)
        uxy = self.filter_func(X * Y, **self.filter_args)
        # compute (weighted) variances and covariances
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        R = data_range
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2
        A1, A2, B1, B2 = ((2 * ux * uy + C1,
                           2 * vxy + C2,
                           ux ** 2 + uy ** 2 + C1,
                           vx + vy + C2))
        D = B1 * B2
        S = (A1 * A2) / D

        # to avoid edge effects will ignore filter radius strip around edges
        pad = (self.win_size - 1) // 2

        # compute (weighted) mean of ssim
        mssim = crop(S, pad).mean()
        return S, mssim
