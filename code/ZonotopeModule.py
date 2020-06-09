import torch
import numpy as np
from networks import Normalization, Conv, FullyConnected


def give_bounds(magnitude_matrix, center_tensor):
    absolute_values = torch.sum(magnitude_matrix.clone().abs().detach(), 0).unsqueeze(0)
    lower_bounds = center_tensor - absolute_values
    upper_bounds = center_tensor + absolute_values
    lb = lower_bounds[0]
    ub = upper_bounds[0]

    return lb, ub


def conv2d_zono(inp, layer, weight):
    if layer.padding_mode == 'circular':
        expanded_padding = (
            (layer.padding[1] + 1) // 2, layer.padding[1] // 2, (layer.padding[0] + 1) // 2, layer.padding[0] // 2)
        return torch.nn.functional.conv2d(torch.nn.functional.pad(inp, expanded_padding, mode='circular'), weight, None,
                                          layer.stride, layer._pair(0), layer.dilation, layer.groups)
    return torch.nn.functional.conv2d(inp, weight, None, layer.stride, layer.padding, layer.dilation, layer.groups)


class ZonotopeTransformer(torch.nn.Module):
    def __init__(self, slopes_list, name):
        super(ZonotopeTransformer, self).__init__()
        self.slopes = torch.nn.ParameterList([])
        for s in slopes_list:
            self.slopes.append(torch.nn.Parameter(s))
        self.count_relu = len(slopes_list)
        self.name = name

    def iterate_network(self, net, magnitude_matrix, center_tensor):

        curr_relu = 0
        for layer in net.layers:


            if isinstance(layer, Normalization):

                center_tensor = layer.forward(center_tensor)
                magnitude_matrix = magnitude_matrix / layer.sigma[0][0][0][0]

            elif isinstance(layer, torch.nn.Linear):

                magnitude_matrix = torch.nn.functional.linear(magnitude_matrix, layer.weight)
                center_tensor = layer.forward(center_tensor)

            elif isinstance(layer, torch.nn.Flatten):

                center_tensor = layer.forward(center_tensor)

            elif isinstance(layer, torch.nn.ReLU):

                # error_extension = torch.zeros(1, magnitude_matrix.size()[1])
                # magnitude_matrix = torch.cat((magnitude_matrix, error_extension), 0)
                center_size = center_tensor.size()
                num_col = magnitude_matrix.shape[1]
                center_tensor = center_tensor.view((1, num_col))
                if self.name == "conv3" or self.name=="conv4":
                    absolute_values = torch.sum(magnitude_matrix.clone().detach().abs(),0).unsqueeze(0)
                else:
                    absolute_values = torch.sum(magnitude_matrix.clone().abs(),0).unsqueeze(0)

                lower_bounds = center_tensor - absolute_values
                upper_bounds = center_tensor + absolute_values
                lx = lower_bounds[0]
                ux = upper_bounds[0]

                basic_slopes = ux / (ux - lx)
                ind_zero = np.argwhere(ux <= 0).flatten()


                ind_crossing = np.argwhere(np.logical_and(ux>0,lx<=0))[0]


                if ind_zero.size()[0] > 0:
                    magnitude_matrix[:, ind_zero] = 0
                    center_tensor[0, ind_zero] = 0
                if ind_crossing.size()[0] > 0:

                    ind_left = []
                    ind_right = []
                    ind_crossing = ind_crossing.detach().numpy()
                    for index in ind_crossing:
                        if 0 <= self.slopes[curr_relu][index] <= basic_slopes[index]:
                            ind_left.append(index)

                        else:
                            ind_right.append(index)


                    ind_right = np.array(ind_right)
                    ind_left = np.array(ind_left)
                    ind_crossing = np.array(ind_crossing)


                    left_slopes = self.slopes[curr_relu][ind_left]
                    right_slopes = self.slopes[curr_relu][ind_right]

                    mag_left = (1 - left_slopes) * ux[ind_left] / 2
                    mag_right = -1 * right_slopes * lx[ind_right] / 2
                    magnitude_matrix[:, ind_crossing] = magnitude_matrix[:, ind_crossing] * (
                        self.slopes[curr_relu][ind_crossing])

                    if ind_left.size >0:
                        left_extensions = torch.zeros((ind_left.shape[0], magnitude_matrix.size()[1]))
                        for i, index in enumerate(ind_left):
                            left_extensions[i, index] = mag_left[i]
                        magnitude_matrix = torch.cat((magnitude_matrix, left_extensions), 0)

                    if ind_right.size>0:
                        right_extensions = torch.zeros((ind_right.shape[0], magnitude_matrix.size()[1]))
                        for i, index in enumerate(ind_right):
                            right_extensions[i, index] = mag_right[i]
                        magnitude_matrix = torch.cat((magnitude_matrix, right_extensions), 0)


                    center_tensor[0, ind_left] = left_slopes * center_tensor[
                        0, ind_left].clone() + mag_left
                    center_tensor[0, ind_right] = right_slopes * center_tensor[
                        0, ind_right].clone() + mag_right

                center_tensor = center_tensor.view(center_size)


                curr_relu += 1

            elif isinstance(layer, torch.nn.Conv2d):
                temp = magnitude_matrix.view(magnitude_matrix.size()[0], center_tensor.size()[1],
                                             center_tensor.size()[2], center_tensor.size()[3])
                y = conv2d_zono(temp, layer, layer.weight)
                norows = magnitude_matrix.size()[0]
                nocols = y.size()[0] * y.size()[1] * y.size()[2] * y.size()[3] / magnitude_matrix.size()[0]
                magnitude_matrix = y.view((norows, int(nocols)))
                center_tensor = layer.forward(center_tensor)

        return magnitude_matrix, center_tensor

    def forward(self, net, magnitude_matrix, center_tensor):
        magnitude_matrix, center_tensor = self.iterate_network(net, magnitude_matrix, center_tensor)
        return magnitude_matrix, center_tensor


class SlopesClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'slopes'):
            slopes_list = module.slopes

            for s in slopes_list:
                ss = s.data
                ss = ss.clamp(0, 1)
                s.data = ss


