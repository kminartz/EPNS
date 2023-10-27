import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import configs
from torch.nn.functional import one_hot
import torch.nn.functional as F

class Config():
    def __init__(self, config_dict):
        self.config_dict = config_dict

def arr_to_im(arr, is_one_hot_enc=False):
    # shape arr: (c, h, w)

    if is_one_hot_enc: #((arr == 0) | (arr == 1)).all()
        # one hot encoded img
        colors = [  # black, blue, green, red, yellow
            np.array([[0.,0.,0.]]), np.array([[0.,0.,1.]]), np.array([[0.,1.,0.]]), np.array([[1.,0.,0.]]),
                  np.array([[1.,1.,0.]])
        ]
        to_plot = np.zeros((*arr.shape[-2:], 3))  #(h, w, c)
        for i in range(arr.shape[-3]):
            to_plot[arr[i] == 1.] += colors[i]
    else:  # if we have !1 and !3 channels, we'll fix it later
        # assert arr.shape[0] == 1 or arr.shape[0] == 3, 'expected either 1 or 3 channels!'
        to_plot = np.moveaxis(arr, (0, 1, 2), (2, 0, 1))
    return to_plot

def plot_cell_image(datapoint, predicted=None, title_append='', no_title=False, show_plot=True):
    datapoint = datapoint[0]
    to_plot = arr_to_im(datapoint)
    plt.imshow(to_plot)
    if not no_title:
        plt.title('datapoint')
    if predicted is not None:
        bar = np.ones_like(to_plot)[:, :3]
        pred = predicted[0]
        to_plot_pred = arr_to_im(pred)
        to_plot = np.concatenate([to_plot, bar, to_plot_pred], axis=1)
        plt.imshow(to_plot)
        if not no_title:
            plt.title('datapoint (left) -- prediction (right) -- ' + title_append)
    if show_plot:
        plt.show()
    return to_plot

def plot_NS_image(datapoint, predicted=None, title_append='', no_title=False, show_plot=True):
    datapoint = datapoint[0]
    to_plot = arr_to_im(datapoint)
    plt.imshow(to_plot)
    if not no_title:
        plt.title('datapoint')
    if predicted is not None:
        bar = np.zeros_like(to_plot)[:, :3]
        pred = predicted[0]
        to_plot_pred = arr_to_im(pred)
        to_plot = np.concatenate([to_plot, bar, to_plot_pred], axis=1)
        if not no_title:
            plt.title('datapoint (left) -- prediction (right) -- ' + title_append)
    if show_plot:
        plt.show()
    return to_plot


def make_gif(list_of_true, list_of_pred, save_path, xlabels=None):
    first_true = list_of_true[0][0]
    # print(first_true.shape)
    bar = np.ones_like(arr_to_im(first_true))[:, :3]
    list_of_conc = [np.concatenate([
        arr_to_im(list_of_true[i][0]), bar, arr_to_im(list_of_pred[i][0])], axis=1) for i, _ in enumerate(list_of_true)]
    fig = plt.figure()
    plt.title('ground truth evolution (left) -- rollout (right)')
    im = plt.imshow(list_of_conc[0])
    if xlabels is not None:
        plt.xlabel(xlabels[0])

    def animate(k):
        im.set_array(list_of_conc[k])
        if xlabels is not None:
            plt.xlabel(xlabels[k])
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=len(list_of_conc))
    if not save_path[-4:] == '.gif':
        anim.save(save_path + '.gif')
    else:
        anim.save(save_path)


def make_gif_NS(list_of_true, list_of_pred, save_path, xlabels=None):
    first_true = list_of_true[0][0]
    # print(first_true.shape)
    bar = np.ones_like(arr_to_im(first_true))[:, :3]
    list_of_conc = [np.concatenate([
        arr_to_im(list_of_true[i][0]), bar, arr_to_im(list_of_pred[i][0])], axis=1) for i, _ in enumerate(list_of_true)]
    fig = plt.figure(figsize=(10,5))
    plt.title('ground truth evolution (left) -- rollout (right)')



    if xlabels is not None:
        plt.xlabel(xlabels[0])

    def animate(k):
        plt.cla()
        vmin = np.min(list_of_conc[k])
        vmax = np.max(list_of_conc[k])
        plt.title('ground truth evolution (left) -- rollout (right)')
        plt.contourf(list_of_conc[k][..., 0], cmap=plt.cm.jet, vmin=vmin, vmax=vmax, levels=25)

    anim = animation.FuncAnimation(fig, animate, frames=len(list_of_conc))
    if not save_path[-4:] == '.gif':
        anim.save(save_path + '.gif')
    else:
        anim.save(save_path)


def get_output_shape(model, input_dim):
    if isinstance(input_dim, torch.Size):
        return model(torch.rand(*(input_dim))).data.shape
    else:
        return model(*(torch.rand(*(dim)) for dim in input_dim)).data.shape

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def from_time_batch_to_batch_time(time_sorted_list_of_tensors):
    l = time_sorted_list_of_tensors
    num_samples = l[0].shape[0] # batch size
    out = []
    for i in range(num_samples):
        out_i = []
        for step in range(len(l)):
            out_i.append(l[step][i:i+1])
        out.append(out_i)
    return out

def create_circular_mask(h, w, center, radius):  # center in (x,y)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center < radius
    return mask

def load_config(cfg_str, state_dict=None):
    cfg = getattr(configs, cfg_str)
    cfg_dict = dict(cfg.config_dict)

    if state_dict is not None:  # replace state dict str with new state dict
        old = cfg_dict['experiment']['state_dict_fname']
        new_state_dict_pointer = {'state_dict_fname': state_dict}
        cfg_dict['experiment'] = new_state_dict_pointer
        print(f'changed state dict from {old} \t\t\t to {state_dict} \t\t\t in {cfg_str}!')

    cfg = Config(config_dict=cfg_dict)
    return cfg

def get_three_rotation_matrices(get_identity_rotation_matrices=False, rotate_only_2d=False):
    import math
    if not get_identity_rotation_matrices:
        degree = math.pi * np.random.uniform(-180, 180) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix1 = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        degree = math.pi * np.random.uniform(-180, 180) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix2 = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        degree = math.pi * np.random.uniform(-180, 180) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix3 = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        if rotate_only_2d:
            return [torch.Tensor(matrix3).double()]
        else:
            return [torch.Tensor(matrix1).double(), torch.Tensor(matrix2).double(), torch.Tensor(matrix3).double()]
    else:
        return [torch.eye(3).double() for _ in range(3)]


def get_permutation_idx(get_identity_permutation=False, num_elements=5):
    import random
    r = [e for e in range(num_elements)]
    if not get_identity_permutation:
        random.shuffle(r)
    return tuple(r)



class PCAException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.pnts_centered = None

def make_dict_serializable(config):
    import json
    for k, v in config.items():
        try:
            json.dumps(v)
        except:
            config[k] = str(v)
    return config

def get_onehot_grid(tensor, num_classes=None):
    output_numpy = False
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
        output_numpy = True

    if num_classes is None:
        num_classes = -1
    tensor_permuted = torch.movedim(tensor, source=(1), destination=(-1))  # move channel to last axis
    tensor_permuted_onehot = one_hot(tensor_permuted.long(), num_classes)[..., 0, :]
    tensor_onehot = torch.movedim(tensor_permuted_onehot, source=(-1), destination=(1))
    tensor_onehot = tensor_onehot.float()
    if output_numpy:
        return tensor_onehot.numpy()
    return tensor_onehot

def _crop_Nd(num_spatial_dims, enc_ftrs: torch.Tensor, shape: torch.Tensor):
    if isinstance(shape, torch.Tensor) or isinstance(shape, np.ndarray):
        shape = shape.shape
    s_des = shape[-num_spatial_dims:]
    s_current = enc_ftrs.shape[-num_spatial_dims:]
    # first, calculate preliminary paddings - may contain non-integers ending in .5):
    pad_temp = np.repeat(np.subtract(s_des, s_current) / 2, 2)
    # to break the .5 symmetry to round one padding up and one down, we add a small pos/neg number respectively
    # note this will not impact the case where pad_temp[i] is integer since it is still rounded to that integer
    breaking_arr = np.tile([1, -1], int(len(pad_temp) / 2)) / 1000
    pad = tuple(map(lambda p: int(round(p)), pad_temp + breaking_arr))
    enc_ftrs = F.pad(enc_ftrs, pad)
    return enc_ftrs

def get_id_to_type_dict_util(id_tensor, type_tensor):
    id_to_type_dict = {}
    for cell_id in torch.unique(id_tensor):
        cell_id = int(cell_id)
        type = torch.where(id_tensor == cell_id, type_tensor, torch.Tensor([0]).to(id_tensor))  # simply put type 0 for the type where this cell is not
        # get the type as a scalar
        unique_type = torch.max(type.flatten(start_dim=1), dim=1)[0]  # get max (is actual type or zero in case this cell id exists nowhere)
        assert torch.bitwise_or(type == unique_type.view(-1,1,1,1), type == 0).all(), 'should only find 0 and the type in type tensor!'
        id_to_type_dict[cell_id] = unique_type.long()
    return id_to_type_dict

def postprocess_and_discretize_util(out, x, id_to_type_dict=None):
    # nothing fancy (yet), simply get the highest prob of the softmax and discretize
    if id_to_type_dict is None:
        id_to_type_dict = get_id_to_type_dict_util(x[:, 0:1], x[:, 1:2])
    # which cells were present in the model input
    # cells_present should now have shape (bs, cell_ids_present_per_batch_element)
    # hardcode cells that are not present at the start to 0
    # we only add the one-hot probs for channels where cells were already present!
    pred = out
    # pred_disc = torch.max(pred, dim=1, keepdim=True)[1]  # highest logit for a cell
    pred_disc = torch.max(pred, dim=1, keepdim=True)[1]  # simply take MLE sample per pixel
    # pred_disc has shape(bs, 1, h, w), where 1 is a single channel containing the cell ID. so this is not one-hot encoded anymore!
    # change pred shape to (bs, 1, h, w, one_hot_dim)
    type_disc = torch.zeros_like(pred_disc)
    for k, v in id_to_type_dict.items():
        type_disc += v.view(-1,1,1,1) * (pred_disc == k)  # wherever the cell equals a certain ID, the disc type is the type of that cell id
    pred_disc = torch.cat([pred_disc, type_disc], dim=1)  # shape (bs, 2, h, w)

    return pred, pred_disc, id_to_type_dict


def set_size_plots(width, fraction=1, h_to_w_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.
    source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27


    if h_to_w_ratio is None:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2
        ratio = golden_ratio
    else:
        ratio = h_to_w_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def export_legend(legend, filename="legend.png"):
    """
    source: https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
    :param legend:
    :param filename:
    :return:
    """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def put_legend_above_fig(**legend_kwargs):
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), **legend_kwargs)