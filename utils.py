#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import torch
import numpy as np
import operator
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.special as ts

from scipy import interpolate
from functools import reduce




def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({s})
    '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("=" * 50)
        print(message)
        print("=" * 50)




def get_num_params(model):
    '''
    a single entry in cfloat and cdouble count as two parameters
    see https://github.com/pytorch/pytorch/issues/57518
    '''
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = 0
    # for p in model_parameters:
    #     # num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
    #     num_params += p.numel() * (1 + p.is_complex())
    # return num_params

    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))  #### there is complex weight
    return c



### x: list of tensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        return self.x[item]


# whether need to transpose
def plot_heatmap(
    x, y, z, path=None, vmin=None, vmax=None,cmap=None,
    title="", xlabel="x", ylabel="y",show=False
):
    '''
    Plot heat map for a 3-dimension data
    '''
    plt.cla()
    # plt.figure()
    xx = np.linspace(np.min(x), np.max(x))
    yy = np.linspace(np.min(y), np.max(y))
    xx, yy = np.meshgrid(xx, yy)

    vals = interpolate.griddata(np.array([x, y]).T, np.array(z),
        (xx, yy), method='cubic')
    vals_0 = interpolate.griddata(np.array([x, y]).T, np.array(z),
        (xx, yy), method='nearest')
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]

    if vmin is not None and vmax is not None:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic",cmap=cmap,
                vmin=vmin, vmax=vmax,origin='lower')
    elif vmin is not None:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic",cmap=cmap,
                vmin=vmin,origin='lower')
    else:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],cmap=cmap,
                aspect="equal", interpolation="bicubic",origin='lower')
    fig.axes.set_autoscale_on(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()

import contextlib

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class TorchQuantileTransformer():
    '''
    QuantileTransformer implemented by PyTorch
    '''

    def __init__(
            self,
            output_distribution,
            references_,
            quantiles_,
            device=torch.device('cpu')
    ) -> None:
        self.quantiles_ = torch.Tensor(quantiles_).to(device)
        self.output_distribution = output_distribution
        self._norm_pdf_C = np.sqrt(2 * np.pi)
        self.references_ = torch.Tensor(references_).to(device)
        BOUNDS_THRESHOLD = 1e-7
        self.clip_min = self.norm_ppf(torch.Tensor([BOUNDS_THRESHOLD - np.spacing(1)]))
        self.clip_max = self.norm_ppf(torch.Tensor([1 - (BOUNDS_THRESHOLD - np.spacing(1))]))

    def norm_pdf(self, x):
        return torch.exp(-x ** 2 / 2.0) / self._norm_pdf_C

    @staticmethod
    def norm_cdf(x):
        return ts.ndtr(x)

    @staticmethod
    def norm_ppf(x):
        return ts.ndtri(x)

    def transform_col(self, X_col, quantiles, inverse):
        BOUNDS_THRESHOLD = 1e-7
        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = self.norm_cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~torch.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        torch_interp = Interp1d()
        X_col_out = X_col.clone()
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col_out[isfinite_mask] = 0.5 * (
                    torch_interp(quantiles, self.references_, X_col_finite)
                    - torch_interp(-torch.flip(quantiles, [0]), -torch.flip(self.references_, [0]), -X_col_finite)
            )
        else:
            X_col_out[isfinite_mask] = torch_interp(self.references_, quantiles, X_col_finite)

        X_col_out[upper_bounds_idx] = upper_bound_y
        X_col_out[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col_out = self.norm_ppf(X_col_out)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    X_col_out = torch.clip(X_col_out, self.clip_min, self.clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col_out

    def transform(self, X, inverse=True,component='all'):
        X_out = torch.zeros_like(X, requires_grad=False)
        for feature_idx in range(X.shape[1]):
            X_out[:, feature_idx] = self.transform_col(
                X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
            )
        return X_out

    def to(self,device):
        self.quantiles_ = self.quantiles_.to(device)
        self.references_ = self.references_.to(device)
        return self


'''
    Simple normalization layer
'''
class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]





'''
    Simple pointwise normalization layer, all data must contain the same length, used only for FNO datasets
    X: B, N, C
'''
class PointWiseUnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=False)
        self.std = X.std(dim=0, keepdim=False) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                X = X.view(-1, self.mean.shape[0],self.mean.shape[1])   ### align shape for flat tensor
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                X = X.view(-1, self.mean.shape[0],self.mean.shape[1])
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]


'''
    x: B, N (not necessary sorted)
    y: B, N, C (not necessary sorted)
    xnew: B, N (sorted)
'''
def binterp1d(x, y, xnew, eps=1e-9):
    x_, x_indice = torch.sort(x,dim=-1)
    y_ = y[torch.arange(x_.shape[0]).unsqueeze(1),x_indice]

    x_, y_, xnew = x_.contiguous(), y_.contiguous(), xnew.contiguous()

    ind = torch.searchsorted(x_, xnew)
    ind -= 1
    ind = torch.clamp(ind, 0, x_.shape[1] - 1 - 1)
    ind = ind.unsqueeze(-1).repeat([1, 1, y_.shape[-1]])
    x_ = x_.unsqueeze(-1).repeat([1, 1, y_.shape[-1]])

    slopes = ((y_[:, 1:]-y_[:, :-1])/(eps + (x_[:, 1:]-x_[:, :-1])))

    y_sel = torch.gather(y_, 1, ind)
    x_sel = torch.gather(x_,1, ind)
    slopes_sel = torch.gather(slopes, 1, ind)

    ynew =y_sel + slopes_sel * (xnew.unsqueeze(-1) - x_sel)

    return ynew



