import numpy as np
import numpy as n
import cv2
import scipy.interpolate
import scipy.ndimage

def coordinates_to_heatmap_vec(coord):
        heatmap_vec = np.zeros(1024)
        [x1, y1, x2, y2] = coord
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                index = y*32 + x
                heatmap_vec[index] = 1.0   #random.uniform(0.8, 1)
        return heatmap_vec


def heatmap_vec_to_heatmap(heatmap_vec):
    size = 32
    heatmap= np.zeros((size, size))
    for y in range(0, size):
        for x in range(0, size):
            index = y*size + x
            heatmap[y][x] = heatmap_vec[index]
    return heatmap


def average_heatmap_sets(heatmaps_orig, heatmaps_flip):
    heatmaps = []
    for ith_map, heatmap_orig in enumerate(heatmaps_orig):
        heatmap_flip = heatmaps_flip[ith_map]
        heatmap = average_two_heatmaps(heatmap_orig, heatmap_flip)
        heatmaps.append(heatmap)
    return heatmaps


def average_two_heatmaps(heatmap1, heatmap2):
    return (heatmap1 + heatmap2) / 2


def average_multiple_heatmap_sets(heatmaps_from_multi_res):
    avg_heatmaps = []
    num_of_res = len(heatmaps_from_multi_res)
    for ith_map in range(15):
        heatmap_total = np.zeros((64, 64))
        for heatmaps_from_one_res in heatmaps_from_multi_res:
            heatmap = heatmaps_from_one_res[ith_map]
            heatmap_total += heatmap
        avg_heatmap = heatmap_total / num_of_res
        avg_heatmaps.append(avg_heatmap)
    return avg_heatmaps


def get_central_heatmaps(heatmaps, ratio, norm_size):
    # crop heatmaps by 1/scale
    st = int(norm_size * (1 - ratio)/2)
    ed = int(norm_size * (1 + ratio)/2)
    heatmaps_cropped = [heatmap[st:ed, st:ed] for heatmap in heatmaps]

    # get the middle heatmaps of norm size
    heatmaps_output = []
    for heatmap_cropped in heatmaps_cropped:
        heatmap_central = cv2.resize(heatmap_cropped, (norm_size, norm_size))
        heatmaps_output.append(heatmap_central)

    return heatmaps_output


def pad_heatmaps(heatmaps, norm_size, scale):
    ht_resize = int(norm_size * scale)
    wid_resize = int(norm_size * scale)
    pad_num = int(norm_size * (1 - scale) / 2)
    heatmaps_pad = []
    heatmaps_np = np.asarray(heatmaps)
    for heatmap_np in heatmaps_np:
        heatmap_crop = congrid(heatmap_np,
                               (ht_resize, wid_resize),
                               method = 'linear',
                               centre = False,
                               minusone = False)

        if wid_resize + pad_num * 2 == norm_size:
            heatmap_pad = np.lib.pad(heatmap_crop,
                                     ((pad_num, pad_num), (pad_num, pad_num)),
                                     'constant',
                                     constant_values = (0, 0))
        elif scale in [0.4, 0.65, 0.75, 0.8, 0.9]:
            heatmap_pad = np.lib.pad(heatmap_crop,
                                     ((pad_num , pad_num + 1), (pad_num , pad_num + 1)), #scale = 0.8 or 0.65 or 0.75
                                     #((pad_num + 1, pad_num + 1), (pad_num + 1, pad_num + 1)), #scale= 0.7 or 0.6
                                     'constant',
                                     constant_values = (0, 0))
        else: # scale in [0.6, 0.7]
            heatmap_pad = np.lib.pad(heatmap_crop,
                                     #((pad_num , pad_num + 1), (pad_num , pad_num + 1)), #scale = 0.8 or 0.65 or 0.75
                                     ((pad_num + 1, pad_num + 1), (pad_num + 1, pad_num + 1)), #scale= 0.7 or 0.6
                                     'constant',
                                     constant_values = (0, 0))


        heatmaps_pad.append(heatmap_pad)
    return heatmaps_pad


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [n.float64, n.float32]:
        a = n.cast[float](a)

    m1 = n.cast[int](minusone)
    ofs = n.cast[int](centre) * 0.5
    old = n.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = n.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = n.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = n.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = n.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = n.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = n.mgrid[nslices]

        newcoords_dims = range(n.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None


def rebin(np_array, shape):
    print(shape)
    print(np_array.shape)
    sh = shape[0], np_array.shape[0]//shape[0], shape[1], np_array.shape[1]//shape[1]
    return np_array.reshape(sh).mean(-1).mean(1)
