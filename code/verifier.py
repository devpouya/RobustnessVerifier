import argparse

import torch
import numpy as np

from networks import Normalization, Conv, FullyConnected
from ZonotopeModule import  give_bounds, SlopesClipper, ZonotopeTransformer

DEVICE = 'cpu'
INPUT_SIZE = 28


def verify_bounds(magnitude_matrix, center_tensor, true_label):
    lowerBounds, upperBounds = give_bounds(magnitude_matrix, center_tensor)
    upperBounds[true_label] = lowerBounds[true_label]
    a, ind = torch.max(upperBounds, 0)
    return ind.item() == true_label


def verify_same_eps(true_label, img, sumsMatrix):
    for i in range(0, sumsMatrix.size()[1]):
        if i != true_label:
            imgs = img[0, true_label] - img[0, i]
            diffs = sumsMatrix[:, true_label] - sumsMatrix[:, i]
            absV = torch.sum(diffs.abs())
            if (imgs - absV <= 0).item():
                return False

    return True



def meanDeviationLoss(magnitude_matrix, center_tensor, true_label, name):
    if name == "conv3" or name=="conv4":
        absolute_values = torch.sum(magnitude_matrix.clone().detach().abs(),0).unsqueeze(0)
    else:
        absolute_values = torch.sum(magnitude_matrix.clone().abs(),0).unsqueeze(0)

    lower_bounds = center_tensor - absolute_values
    upper_bounds = center_tensor + absolute_values
    lb = lower_bounds[0]
    ub = upper_bounds[0]
    target = lb[true_label]
    ub[true_label] = lb[true_label]
    loss = torch.sum(ub - target)
    ub[true_label] = target - 1
    loss += torch.sum(ub == target)
    return loss / (lb.shape[0] - 1)


def conv2d_zono(inp, layer, weight):
    return torch.nn.functional.conv2d(inp, weight, None, layer.stride, layer.padding, layer.dilation, layer.groups)


def iterate_network(net, magnitude_matrix, center_tensor):
    slopes_layers = []
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

            center_size = center_tensor.size()
            num_col = magnitude_matrix.shape[1]
            center_tensor = center_tensor.view((1, num_col))

            lx, ux = give_bounds(magnitude_matrix, center_tensor)

            basic_slopes = ux / (ux - lx)

            slopes_param = basic_slopes.clone().detach().requires_grad_(True).type(torch.float32)
            slopes_layers.append(slopes_param)

            ind_zero = np.argwhere(ux <= 0).flatten()
            ind_keep = np.argwhere(lx > 0).flatten()
            all_ind = np.arange(num_col)
            ind_ignore = np.union1d(ind_zero, ind_keep)
            if ind_ignore is not None or not np.array_equal(all_ind, ind_ignore):
                ind_crossing = np.setdiff1d(all_ind, ind_ignore)
            elif np.array_equal(ind_ignore, all_ind):
                ind_crossing = None
            else:
                ind_crossing = all_ind


            if ind_zero.size()[0] > 0:
                magnitude_matrix[:, ind_zero] = 0
                center_tensor[0, ind_zero] = 0

            if ind_crossing is not None and ind_crossing.size != 0:
                slopes = basic_slopes[ind_crossing]
                error_extension = torch.zeros(ind_crossing.shape[0], magnitude_matrix.size()[1])

                mag_left = (1 - slopes) * ux[ind_crossing] / 2
                magnitude_matrix[:, ind_crossing] = magnitude_matrix[:, ind_crossing] * (
                    slopes)
                for i, index in enumerate(ind_crossing):
                    error_extension[i,index] = mag_left[i]
                magnitude_matrix = torch.cat((magnitude_matrix, error_extension), 0)
                center_tensor[0, ind_crossing] = center_tensor[
                    0, ind_crossing].clone() * slopes + mag_left
            center_tensor = center_tensor.view(center_size)

        elif isinstance(layer, torch.nn.Conv2d):
            temp = magnitude_matrix.view(magnitude_matrix.size()[0], center_tensor.size()[1],
                                         center_tensor.size()[2], center_tensor.size()[3])
            y = conv2d_zono(temp, layer, layer.weight)
            norows = magnitude_matrix.size()[0]
            nocols = y.size()[0] * y.size()[1] * y.size()[2] * y.size()[3] / magnitude_matrix.size()[0]
            magnitude_matrix = y.view((norows, int(nocols)))
            center_tensor = layer.forward(center_tensor)

    return magnitude_matrix, center_tensor, slopes_layers


def clip_input(img, eps, sumsMatrix):
    img_flat = torch.flatten(img)
    ind_left = np.argwhere(img_flat + eps > 1)
    ind_right = np.argwhere(img_flat - eps < 0)
    x_left = (img_flat[ind_left] - eps + 1) / 2
    x_right = (img_flat[ind_right] + eps) / 2
    eps_left = 1 - x_left
    sumsMatrix[ind_left, ind_left] = eps_left
    sumsMatrix[ind_right, ind_right] = x_right
    img_flat[ind_left] = x_left
    img_flat[ind_right] = x_right
    img = img_flat.view(img.size())

    return img, sumsMatrix


def analyze(net, inputs, eps, true_label, name):

    (img_dim0, img_dim1, img_dim2, img_dim3) = inputs.shape
    input_size = img_dim2 * img_dim3
    magnitude_matrix_init = eps * torch.eye(input_size)

    center_tensor_init = inputs
    torch.autograd.set_detect_anomaly(True)
    center_tensor_init, magnitude_matrix_init = clip_input(center_tensor_init, eps, magnitude_matrix_init)
    magnitude_matrix, center_tensor, slopes_init = iterate_network(net, magnitude_matrix_init, center_tensor_init)

    ver = verify_bounds(magnitude_matrix, center_tensor, true_label)
    if ver:
        return True

    else:
        ken_ver = verify_same_eps(true_label, center_tensor, magnitude_matrix)
        if ken_ver:
            return True

        if isinstance(net, Conv):
            max_iter = 100
            lr = 0.01
        else:
            max_iter = 1000
            lr = 0.01

        zonoModel = ZonotopeTransformer(slopes_init,name)
        slopesClipper = SlopesClipper()
        optimizer = torch.optim.SGD(zonoModel.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)

        n_iter = 0

        while n_iter < max_iter:
            zonoModel.apply(slopesClipper)

            magnitude_matrix, center_tensor = zonoModel(net, magnitude_matrix_init, center_tensor_init)

            loss = meanDeviationLoss(magnitude_matrix, center_tensor, true_label, name)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            zonoModel.apply(slopesClipper)
            scheduler.step(loss)

            ver = verify_bounds(magnitude_matrix, center_tensor, true_label)
            if ver:
                return True
            else:
                ken_ver = verify_same_eps(true_label, center_tensor, magnitude_matrix)
                if ken_ver:
                    return True
            n_iter += 1

        ver = verify_bounds(magnitude_matrix, center_tensor, true_label)
        ken_ver = verify_same_eps(true_label, center_tensor, magnitude_matrix)

        return ken_ver or ver



def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label, args.net):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
