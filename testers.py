import numpy as np
import torch
from numpy import linalg as LA

import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

def test_irregular_sparsity(model):
    """

        :param model: saved re-trained model
        :return:
        """

    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if "bias" in name:
            continue
        zeros = np.sum(weight.cpu().detach().numpy() == 0)
        total_zeros += zeros
        non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
        total_nonzeros += non_zeros
        zeros = np.sum(weight.cpu().detach().numpy() == 0)
        non_zero = np.sum(weight.cpu().detach().numpy() != 0)
        print("irregular zeros: {}, irregular sparsity is: {:.4f}".format(zeros, zeros / (zeros + non_zero)))

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros+total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")


def test_column_sparsity(model):
    """

    :param model: saved re-trained model
    :return:
    """

    total_zeros = 0
    total_nonzeros = 0

    total_column = 0
    total_empty_column = 0

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4): # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            weight2d = weight.reshape(weight.shape[0], -1)
            column_num = weight2d.shape[1]

            empty_column = np.sum(np.sum(weight2d.cpu().detach().numpy(), axis=0) == 0)
            print("(total/empty) column of {} is: ({}/{}). column sparsity is: {:.4f}".format(
                name, weight.size()[1]*weight.size()[2]*weight.size()[3], empty_column, empty_column / column_num))

            total_column += column_num
            total_empty_column += empty_column
    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("total number of column: {}, empty-column: {}, column sparsity is: {:.4f}".format(
        total_column, total_empty_column, total_empty_column / total_column))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros)/total_nonzeros))

    # unused = calculate_unused_weight(model)
    # print("only consider conv layers, including unused weight, compression rate is: {:.4f}".format(
    #     (total_zeros + total_nonzeros) / (total_nonzeros - unused)))
    # print("===========================================================================\n\n")

def test_channel_sparsity(model):
    """

    :param model: saved re-trained model
    :return:
    """

    total_zeros = 0
    total_nonzeros = 0

    total_channel = 0
    total_empty_channel = 0

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4):
            weight = weight.cpu().detach().numpy()
            """ check channel sparsity based on column sparsity"""
            zeros = np.sum(weight == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight != 0)
            total_nonzeros += non_zeros

            channel_num = weight.shape[1]
            empty_channel = 0
            for i in range(channel_num):
                # print(np.sum(weight[:,i,:,:].cpu().detach().numpy()))
                if np.sum(weight[:,i,:,:]) == 0:
                    empty_channel += 1
            print("(total/empty) channel of {} is: ({}/{}). channel sparsity is: {:.4f}".format(
                name, weight.shape[1], empty_channel, empty_channel / channel_num))
            total_channel += channel_num
            total_empty_channel += empty_channel

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("total number of channel: {}, empty-channel: {}, channel sparsity is: {:.4f}".format(
        total_channel, total_empty_channel, total_empty_channel / total_channel))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))

    # unused = calculate_unused_weight(model)
    # print("only consider conv layers, including unused weight, compression rate is: {:.4f}".format(
    #     (total_zeros + total_nonzeros) / (total_nonzeros - unused)))
    # print("===========================================================================\n\n")

            # almost_empty_channel = 0
            # for i in range(weight2d.size()[1]):
            #     channel_i = weight2d[0, i, :]
            #     # print(channel_i)
            #     zeros = np.sum(channel_i.cpu().detach().numpy() == 0)
            #     channel_empty_ratio = zeros / weight2d.size()[2]
            #     if channel_empty_ratio == 1:
            #         almost_empty_channel += 1
            #     # print(zeros, weight2d.size()[2])
            #     # print(channel_empty_ratio)
            # print("({} {}) almost empty channel: {}, total channel: {}. ratio: {}%".format(name, weight.size(),
            #     almost_empty_channel, weight2d.size()[1], 100.0 * almost_empty_channel / weight2d.size()[1]))


def test_filter_sparsity(model):
    """

    :param model: saved re-trained model
    :return:
    """

    total_zeros = 0
    total_nonzeros = 0

    total_filters = 0
    total_empty_filters = 0

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4): # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            empty_filters = 0
            filter_num = weight.size()[0]

            for i in range(filter_num):
                if np.sum(weight[i,:,:,:].cpu().detach().numpy()) == 0:
                    empty_filters += 1
            print("(total/empty) filter of {} is: ({}/{}). filter sparsity is: {:.4f}".format(
                name, weight.size()[0], empty_filters, empty_filters / filter_num))

            total_filters += filter_num
            total_empty_filters += empty_filters

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("total number of filters: {}, empty-filters: {}, filter sparsity is: {:.4f}".format(
        total_filters, total_empty_filters, total_empty_filters / total_filters))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))

    # unused = calculate_unused_weight(model)
    # print("only consider conv layers, including unused weight, compression rate is: {:.4f}".format(
    #     (total_zeros + total_nonzeros) / (total_nonzeros - unused)))
    # print("===========================================================================\n\n")

def test_filter_balance(model):
    """

    :param model: saved re-trained model
    :return:
    """

    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4 and "shortcut" not in name): # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
            kernel_num = weight3d.shape[1]

            # print(kernel_num)
            for i in range(weight3d.shape[0]):
                empty_kernel_num = 0
                for j in range(weight3d.shape[1]):
                    # print(weight3d[i,j,:])
                    if(np.sum(weight3d[i,j,:].cpu().detach().numpy()) == 0):
                        empty_kernel_num += 1
                print(kernel_num, empty_kernel_num)


def test_pattern_distribution(model):
    """

    :param model: saved re-trained model
    :return:
    """
    pattern1 = [0,2,4,6,8]
    pattern2 = [0,2,3,5,6]
    pattern3 = [0,1,2,7,8]
    pattern4 = [2,3,5,6,8]
    pattern5 = [0,1,6,7,8]
    pattern6 = [0,2,3,5,8]
    pattern7 = [1,2,6,7,8]
    pattern8 = [0,3,5,6,8]
    pattern9 = [0,1,2,6,7]
    pattern10 = [1,2,5,6,8]
    pattern11 = [0,5,6,7,8]
    pattern12 = [0,2,3,6,7]
    pattern13 = [0,1,2,3,8]
    pattern14 = [0,1,3,6,8]
    pattern15 = [0,1,2,5,6]
    pattern16 = [0,2,5,7,8]
    pattern17 = [2,3,6,7,8]
    pattern18 = [0,2,6,7,8]
    pattern19 = [0,2,3,6,8]
    pattern20 = [0,1,2,6,8]
    pattern21 = [0,2,5,6,8]

    patterns_dict = {1: pattern1,
                     2: pattern2,
                     3: pattern3,
                     4: pattern4,
                     5: pattern5,
                     6: pattern6,
                     7: pattern7,
                     8: pattern8,
                     9: pattern9,
                     10: pattern10,
                     11: pattern11,
                     12: pattern12,
                     13: pattern13,
                     14: pattern14,
                     15: pattern15,
                     16: pattern16,
                     17: pattern17,
                     18: pattern18,
                     19: pattern19,
                     20: pattern20,
                     21: pattern21
                     }

    """initialize distribution"""
    model_pattern_distribution = {}
    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4):
            model_pattern_distribution[name] = {}

    for name, weight in model.named_parameters(): # loop layer
        if(len(weight.size()) == 4):
            weight3d = (weight.reshape(weight.shape[0], weight.shape[1], -1)).cpu().detach().numpy()
            layer_pattern_distribute = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0}
            print("layer {}: each filter has pattern distributed in: ".format(name))
            for i in range(weight3d.shape[0]): # loop filter
                filter_pattern_distribute = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0}
                for j in range(weight3d.shape[1]): # loop kernel
                    pattern = np.where(weight3d[i,j,:] == 0)[0]
                    for k, v in patterns_dict.items():
                        pattern2compare = np.array(v)
                        if np.array_equal(pattern, pattern2compare):
                            layer_pattern_distribute[k] += 1
                            filter_pattern_distribute[k] += 1
                print("layer {}, filter {} --> {}".format(name, i, filter_pattern_distribute))
            print("-------------------\nthe total pattern distribution in this layer -> {} is: ".format(name))
            print(layer_pattern_distribute, "\n\n")
            model_pattern_distribution[name] = layer_pattern_distribute


    np.save("pattern_dict.npy", model_pattern_distribution)


def calculate_unused_weight(model):
    """
        helper funtion to calculate the corresponding filter add-on sparsity to next layer empty channel

        :param model: saved re-trained model
        :return:
        """

    weight_dict = {} # indexed weight copy
    m = 1 # which layer
    n = 1 # which layer
    counter = 1 # layer counter
    total_unused_number = 0 # result
    flag1 = False # detect sparsity type
    flag2 = False # detect sparsity type

    for name, weight in model.named_parameters(): # calculate total layer
        if (len(weight.size()) == 4 and "downsample" not in name):
            weight_dict[counter] = weight
            counter += 1
    counter = counter - 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            if m != 1:
                empty_channel_index = []
                for i in range(weight3d.size()[1]):
                    non_zero_filter = np.where(weight3d[:, i, :].cpu().detach().numpy().any(axis=1))[0]
                    if non_zero_filter.size == 0:
                        channel_i = weight3d[0, i, :]
                    else:
                        channel_i = weight3d[non_zero_filter[0], i, :]
                    zeros = np.sum(channel_i.cpu().detach().numpy() == 0)
                    channel_empty_ratio = zeros / weight3d.size()[2]
                    if channel_empty_ratio == 1:
                        empty_channel_index.append(i)
                        flag1 = True
                # print(name, empty_channel_index)

                previous_layer = weight_dict[m - 1]
                filter_unused_num = 0
                for filter_index in empty_channel_index:
                    target_filter = previous_layer[filter_index, :, :, :]
                    filter_unused_num += np.sum(target_filter.cpu().detach().numpy() != 0)  # != 0 to calculate sparsity
                total_unused_number += filter_unused_num

            m += 1

            #=====================================================================================#
            """ calculate unused channel, empty filter by filter pruning => next layer channel """
            if n != counter:
                empty_filter_index = []
                for j in range (weight.size()[0]):
                    if np.sum(weight[j, :, :, :].cpu().detach().numpy()) == 0:
                        empty_filter_index.append(j)
                        flag2 = True
                # print(empty_filter_index)
                next_layer = weight_dict[n + 1]
                channel_unused_num = 0
                for channel_index in empty_filter_index:
                    target_channel = next_layer[:, channel_index, :, :]
                    channel_unused_num += np.sum(target_channel.cpu().detach().numpy() != 0)  # != 0 to calculate sparsity
                total_unused_number += channel_unused_num

            n += 1

    if flag1 and not flag2:
        print("your model has column sparsity")
    elif flag2 and not flag1:
        print("your model has filter sparsity")
    elif flag1 and flag2:
        print("your model has column AND filter sparsity")
    elif not flag1 and not flag2:
        print("your model doesn't have redundent weights")
    print("total unused weight number (column => prev filter / filter => next column): ", total_unused_number)
    return total_unused_number



def remove_unused_weights(model):
    weight_dict = {}  # indexed weight copy
    # ------ column & filter prune -----------------
    channel_to_remove = {}
    filter_to_remove = {}  # index of the filters need to be removed in previous layer

    # ------ check L2-norm -------------------------
    L2_norms_8col = []
    L2_norms_all = []
    L2_8_dict = {}
    L2_all_dict = {}

    m = 1  # which layer
    n = 1  # which layer
    layer_cont = 1

    total_unused_number = 0  # result

    for name, weight in model.named_parameters():  # calculate total layer
        if (len(weight.size()) == 4 and "downsample" not in name):
            weight_dict[layer_cont] = weight
            layer_cont += 1

    total_layer = layer_cont - 1
    layer_cont = 1
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            if m != 1:  # start from second layer, 1st layer has no previous layer
                # ---------------- Test -----------------
                empty_9col_channel_index = []  # index of channels have 9 empty columns
                empty_8col_channel_index = []  # index of channels have 8 empty columns
                empty_7col_channel_index = []  # index of channels have 7 empty columns
                # ---------------------------------------
                empty_channel_index = []  # index of considered empty channels (9? 8? 7?...)

                for i in range(weight3d.size()[1]):  # loop all channels
                    non_zero_filter = np.where(weight3d[:, i, :].cpu().detach().numpy().any(axis=1))[0]
                    if non_zero_filter.size == 0:
                        channel_i = weight3d[0, i, :]
                    else:
                        channel_i = weight3d[non_zero_filter[0], i, :]
                    zeros = np.sum(channel_i.cpu().detach().numpy() == 0)

                    # ------------ check channel L2-norm ------------
                    # channel2d = weight3d[:, i, :].cpu().detach().numpy()
                    #
                    # column_l2_norm = LA.norm(channel2d, 2, axis=0)
                    # channel_l2_norm = np.sum(column_l2_norm)
                    #
                    # if channel_l2_norm <= 1.5:      # th
                    #     empty_channel_index.append(i)       # find columns under th to prune
                    #     L2_norms_8col.append(channel_l2_norm)   #test
                    # -----------------------------------------------
                    # ---------------- Test -----------------
                    if (True):
                        if (zeros == 7):
                            empty_7col_channel_index.append(i)
                            # print("layer: {} channel {} has {} zeros over {}".format(name, i, zeros, weight3d.size()[2]))
                        if (zeros == 8):
                            empty_8col_channel_index.append(i)
                            # print("layer: {} channel {} has {} zeros over {}".format(name, i, zeros, weight3d.size()[2]))
                        if (zeros == 9):
                            empty_9col_channel_index.append(i)
                            # print("layer: {} channel {} has {} zeros over {}".format(name, i, zeros, weight3d.size()[2]))

                    # ---------------------------------------
                    channel2d = weight3d[:, i, :].cpu().detach().numpy()
                    column_l2_norm = LA.norm(channel2d, 2, axis=0)
                    # channel_l2_norm = np.sum(column_l2_norm)
                    # print(channel_l2_norm)
                    for column_norm in column_l2_norm:
                        if column_norm != 0:
                            L2_norms_all.append(column_norm)

                    if (zeros == 9):
                        empty_channel_index.append(i)

                print("layer: {} 7-zeros channel is: {}/{} ({}%)".format(layer_cont, len(empty_7col_channel_index),
                                                                   weight3d.size()[1], 100.*float(len(empty_7col_channel_index))/float(weight3d.size()[1])))
                print("layer: {} 8-zeros channel is: {}/{} ({}%)".format(layer_cont, len(empty_8col_channel_index),
                                                                   weight3d.size()[1], 100.*float(len(empty_8col_channel_index))/float(weight3d.size()[1])))
                print("layer: {} 9-zeros channel is: {}/{} ({}%)".format(layer_cont, len(empty_9col_channel_index),
                                                                   weight3d.size()[1], 100.*float(len(empty_9col_channel_index))/float(weight3d.size()[1])))
                print("--------------------------------------------")
                L2_8_dict[name] = list(L2_norms_8col)
                #print(L2_norms_8col)
                L2_norms_8col.clear()
                #print("L2_norm_8col")
                #print(L2_norms_8col)
                #print(L2_8_dict[name])

                if layer_cont in channel_to_remove:
                    # print("col before extend in layer {}".format(layer_cont))
                    # print(column_to_remove[layer_cont])
                    channel_to_remove[layer_cont].extend(empty_channel_index)
                    # print("col after extend in layer {}".format(layer_cont))
                    # print(column_to_remove[layer_cont])
                else:
                    channel_to_remove[layer_cont] = empty_channel_index

                if layer_cont - 1 in filter_to_remove:
                    # print("fil before extend in layer {}".format(layer_cont-1))
                    # print(filter_to_remove[layer_cont-1])
                    filter_to_remove[layer_cont - 1].extend(empty_channel_index)
                    # print("fil after extend in layer {}".format(layer_cont-1))
                    # print(filter_to_remove[layer_cont-1])
                else:
                    filter_to_remove[layer_cont - 1] = empty_channel_index

            m += 1

            # =====================================================================================#
            """ calculate unused channel, empty filter by filter pruning => next layer channel """
            if n != total_layer:
                empty_filter_index = []
                distri_dict = {}

                for j in range(weight.size()[0]):
                    if np.sum(weight[j, :, :, :].cpu().detach().numpy()) == 0:
                        empty_filter_index.append(j)
                # print(empty_filter_index)

                total_num = weight[0, :, :, :].cpu().detach().numpy().size  # total weights per filter
                for index in range(weight.size()[0]):
                    zeros = np.count_nonzero(weight[index, :, :, :].cpu().detach().numpy() == 0)    # count zeros in a filter
                    percentage = int(100*zeros/total_num)
                    if percentage in distri_dict:   # get distribution of different percentage
                        distri_dict[percentage] += 1
                    else:
                        distri_dict[percentage] = 1
                print("filter zero percentage in layer: {} (total {} weights)".format(layer_cont, total_num))
                print(sorted(distri_dict.items()))  # print sorted distri_dict
                print("--------------------------------------------")


                # ------ add filters that needs to be removed into dict ---------
                if layer_cont in filter_to_remove:
                    #print("fil before extend in layer {}".format(layer_cont))
                    #print(filter_to_remove[layer_cont])
                    filter_to_remove[layer_cont].extend(empty_filter_index)  # may has duplicated element now
                    #print("fil after extend in layer {}".format(layer_cont))
                    #print(filter_to_remove[layer_cont])
                else:
                    #print("fil before new in layer {}".format(layer_cont))
                    filter_to_remove[layer_cont] = empty_filter_index
                   # print("fil after new in layer {}".format(layer_cont))
                    #print(filter_to_remove[layer_cont])
                # ------ add column that needs to be removed into dict ---------
                if layer_cont + 1 in channel_to_remove:
                    #print("col before extend in layer {}".format(layer_cont + 1))
                   # print(column_to_remove[layer_cont + 1])
                    channel_to_remove[layer_cont + 1].extend(empty_filter_index)  # may has duplicated element now
                    #print("col after extend in layer {}".format(layer_cont + 1))
                   # print(column_to_remove[layer_cont + 1])
                else:
                    #print("col before new in layer {}".format(layer_cont + 1))
                    channel_to_remove[layer_cont + 1] = empty_filter_index
                   # print("col after new in layer {}".format(layer_cont + 1))
                    #print(column_to_remove[layer_cont + 1])

            n += 1
            layer_cont += 1

    # ------------- remove weights ----------------
    layer_cont = 1
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            # ----------- remove filters ----------------
            if layer_cont in filter_to_remove:      #last layer not included
                #print("remove filters in Conv layer: {}".format(layer_cont))
                newWeight = weight.cpu().detach().numpy()
                for index in filter_to_remove[layer_cont]:
                    newWeight[index, :, :, :] = 0  # set unused filter to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ----------- remove channels ---------------
            if layer_cont in channel_to_remove:      #first layer not included
                #print("remove channels in Conv layer: {}".format(layer_cont))
                newWeight = weight.cpu().detach().numpy()
                for index in channel_to_remove[layer_cont]:
                    newWeight[:, index, :, :] = 0  # set unused filter to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            layer_cont += 1

    # plt.figure(1)
    # plt.hist(L2_norms_8col, bins=50, align="mid", rwidth=0.3)
    # plt.title("8 columns L2-norms")
    #
    # plt.figure(2)
    # plt.hist(L2_norms_all, bins=50, align="mid", rwidth=0.3)
    # plt.title("all columns L2-norms")
    # #plt.yscale('log')
    # plt.show()

    # for layer in L2_8_dict:
    #     print(L2_8_dict[layer])

    return channel_to_remove, filter_to_remove

# prune channels that L2 norm smaller than threshold
def post_channel_prune(model, th=0):

         # threshold for channel prune
                    # channels that L2 norm under th will be pruned
    # best th: resnet: 0.3  vgg16: 0.23

    # ------ channels to prune -----------------
    channel_to_remove = {}

    # ------ check L2-norm -------------------------
    L2_norms_channel = []
    L2_dict = {}

    layer_cont = 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            empty_channel_index = []

            for i in range(weight3d.size()[1]):  # loop all channels

                channel2d = weight3d[:, i, :].cpu().detach().numpy()
                columns_l2_norm = LA.norm(channel2d, 2, axis=0)  # [0.1,0.5, ...] 9 columns' L2 norm
                channel_l2_norm = np.sum(columns_l2_norm)  # channel L2 norm sum

                if channel_l2_norm < th:  # th
                    empty_channel_index.append(i)  # find channel under th to prune
                    L2_norms_channel.append(channel_l2_norm)  # test

            # --------------- store satisfied channels in each layer ------------------------
            L2_dict[name] = list(L2_norms_channel)
            L2_norms_channel.clear()

            channel_to_remove[layer_cont] = empty_channel_index

            layer_cont += 1

    for layer in L2_dict:
        print(L2_dict[layer])
    # ------------- remove channels ----------------
    layer_cont = 1
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            if layer_cont in channel_to_remove:
                newWeight = weight.cpu().detach().numpy()
                for index in channel_to_remove[layer_cont]:
                    newWeight[:, index, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            layer_cont += 1

    return th


def post_column_prune(model, th=0):
    # threshold for channel prune
    # channels that L2 norm under th will be pruned
    # best th: resnet: 0.04  vgg16: 0.057 vgg_shortcut:0.0645

    # ------ channels to prune -----------------
    column_to_remove = {}

    # ------ check L2-norm -------------------------
    L2_norms_column = []
    L2_dict = {}

    layer_cont = 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):  # and "shortcut" not in name and "downsample" not in name):
            weight2d = weight.reshape(weight.shape[0], -1)
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            empty_column_index = []

            for i in range(weight2d.size()[1]):  # loop all columns

                column1d = weight2d[:, i].cpu().detach().numpy()
                column_l2_norm = LA.norm(column1d, 2)  # [0.1,0.5, ...] 9 columns' L2 norm

                if column_l2_norm < th:  # th
                    empty_column_index.append(i)  # find channel under th to prune
                    L2_norms_column.append(column_l2_norm)  # test

            # --------------- store satisfied channels in each layer ------------------------
            L2_dict[name] = list(L2_norms_column)
            L2_norms_column.clear()

            column_to_remove[layer_cont] = empty_column_index

            layer_cont += 1

    print("----------- empty columns ------------")
    for layer in L2_dict:
        print(L2_dict[layer])
    # ------------- remove channels ----------------
    layer_cont = 1
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):  # and "shortcut" not in name and "downsample" not in name):
            if layer_cont in column_to_remove:
                shape = weight.shape
                weight = weight.reshape(shape[0], -1)
                for index in column_to_remove[layer_cont]:
                    weight[:, index] = 0  # set channel to 0
            layer_cont += 1

    return th


def post_filter_prune(model, th=0):

           # threshold for channel prune
                    # channels that L2 norm under th will be pruned
    # best th: resnet18: 0.23  vgg16: 0.15

    # ------ channels to prune -----------------
    filter_to_remove = {}

    # ------ check L2-norm -------------------------
    L2_norms_filter = []
    L2_dict = {}

    layer_cont = 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            weight2d = weight.reshape(weight.shape[0], -1)
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """

            empty_filter_index = []

            for i in range(weight2d.size()[0]):  # loop all filters

                filter1d = weight2d[i, :].cpu().detach().numpy()
                filter_l2_norm = LA.norm(filter1d, 2)  # L2 norm of a filter

                if filter_l2_norm < th:  # th
                    empty_filter_index.append(i)  # find channel under th to prune
                    L2_norms_filter.append(filter_l2_norm)  # test

            # --------------- store satisfied filters in each layer ------------------------
            L2_dict[name] = list(L2_norms_filter)
            L2_norms_filter.clear()

            filter_to_remove[layer_cont] = empty_filter_index

            layer_cont += 1

    for layer in L2_dict:
        print(L2_dict[layer])
    # ------------- remove filters ----------------
    layer_cont = 1
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "downsample" not in name):
            if layer_cont in filter_to_remove:  # first layer not included
                newWeight = weight.cpu().detach().numpy()
                for index in filter_to_remove[layer_cont]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            layer_cont += 1

    return th

def post_kernel_prune(model, th=0):
    for name, weight in model.named_parameters():
        weightn = weight.cpu().detach().numpy()
        shape = weightn.shape
        if (len(shape) == 4):  # and "shortcut" not in name):
            weight3d = weightn.reshape(shape[0], shape[1], -1)
            for i in range(shape[0]):  # loop all filters
                filter_i = weight3d[i, :, :]
                kernel_l2_norms = LA.norm(filter_i, 2, axis=1)
                under_threshold = kernel_l2_norms < th
                filter_i[under_threshold, :] = 0
            weight.data = torch.from_numpy(weightn).cuda()
    return th



def test_sparsity(model, column=True, channel=True, filter=True):

    # --------------------- total sparsity --------------------
    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros

    comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)

    if(not column and not channel and not filter):
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

    # --------------------- column sparsity --------------------
    if(column):

        total_column = 0
        total_empty_column = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
                weight2d = weight.reshape(weight.shape[0], -1)
                column_num = weight2d.shape[1]

                empty_column = np.sum(np.sum(np.absolute(weight2d.cpu().detach().numpy()), axis=0) == 0)
                print("(empty/total) column of {}({}) is: ({}/{}). column sparsity is: {:.4f}".format(
                    name, layer_cont, empty_column, weight.size()[1] * weight.size()[2] * weight.size()[3],
                                        empty_column / column_num))

                total_column += column_num
                total_empty_column += empty_column
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of column: {}, empty-column: {}, column sparsity is: {:.4f}".format(
            total_column, total_empty_column, total_empty_column / total_column))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- channel sparsity --------------------
    if (channel):

        total_channels = 0
        total_empty_channels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
                empty_channels = 0
                channel_num = weight.size()[1]

                for i in range(channel_num):
                    if np.sum(np.absolute(weight[:, i, :, :].cpu().detach().numpy())) == 0:
                        empty_channels += 1
                print("(empty/total) channel of {}({}) is: ({}/{}). channel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_channels, weight.size()[1], empty_channels / channel_num))

                total_channels += channel_num
                total_empty_channels += empty_channels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of channels: {}, empty-channels: {}, channel sparsity is: {:.4f}".format(
            total_channels, total_empty_channels, total_empty_channels / total_channels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- filter sparsity --------------------
    if(filter):

        total_filters = 0
        total_empty_filters = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
                empty_filters = 0
                filter_num = weight.size()[0]

                for i in range(filter_num):
                    if np.sum(np.absolute(weight[i, :, :, :].cpu().detach().numpy())) == 0:
                        empty_filters += 1
                print("(empty/total) filter of {}({}) is: ({}/{}). filter sparsity is: {:.4f}".format(
                    name, layer_cont, empty_filters, weight.size()[0], empty_filters / filter_num))

                total_filters += filter_num
                total_empty_filters += empty_filters
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of filters: {}, empty-filters: {}, filter sparsity is: {:.4f}".format(
            total_filters, total_empty_filters, total_empty_filters / total_filters))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

    return comp_ratio


def find_empty_channel_and_filters_resnet18(model):
    conv_weight_dict = {}  # indexed weight copy
    shortcut_weight_dict = {}
    conv_layer_number = []
    shortcut_layer_number = []
    # ------ column & filter prune -----------------
    channel_to_remove = {}
    filter_to_remove = {}  # index of the filters need to be removed in previous layer
    conv_empty_channels = {}
    conv_empty_filters = {}
    sc_empty_channels = {}
    sc_empty_filters = {}  # index of the filters need to be removed in previous layer

    layer_cont = 1
    total_conv_layer = 0
    total_shortcut_layer = 0
    for name, weight in model.named_parameters():  # calculate total layer
        if (len(weight.size()) == 4 and "shortcut" not in name and "downsample" not in name):
            conv_weight_dict[layer_cont] = weight
            conv_layer_number.append(layer_cont)
            total_conv_layer += 1

        if (len(weight.size()) == 4 and ("shortcut" in name or "downsample" in name)):
            shortcut_weight_dict[layer_cont] = weight
            shortcut_layer_number.append(layer_cont)
            total_shortcut_layer += 1

        layer_cont += 1

    total_layer = layer_cont - 1
    layer_cont = 1
    conv_layer_number_index = 0
    for name, weight in model.named_parameters():
        # -------------- Conv layer empty channels ------------------------
        if (len(weight.size()) == 4 and "shortcut" not in name and "downsample" not in name):
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            if conv_layer_number_index != 0:  # start from second layer, 1st layer has no previous layer
                # ---------------------------------------
                empty_channel_index = []  # index of considered empty channels (9? 8? 7?...)

                for i in range(weight.size()[1]):
                    if np.sum(np.absolute(weight[:, i, :, :].cpu().detach().numpy())) == 0:
                        empty_channel_index.append(i)

                conv_empty_channels[layer_cont] = empty_channel_index
                filter_to_remove[conv_layer_number[conv_layer_number_index - 1]] = empty_channel_index
            # =====================================================================================#
            """ calculate unused channel, empty filter by filter pruning => next layer channel """
            if conv_layer_number_index != total_conv_layer - 1:     #not the last layer
                empty_filter_index = []

                for j in range(weight.size()[0]):
                    if np.sum(np.absolute(weight[j, :, :, :].cpu().detach().numpy())) == 0:
                        empty_filter_index.append(j)

                conv_empty_filters[layer_cont] = empty_filter_index
                channel_to_remove[conv_layer_number[conv_layer_number_index + 1]] = empty_filter_index

            conv_layer_number_index += 1

        # -------------- shortcut layer empty channels ------------------------
        if (len(weight.size()) == 4 and (("shortcut" in name) or ("downsample" in name))):
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            # ---------------------------------------
            empty_channel_index = []  # index of considered empty channels (9? 8? 7?...)

            for i in range(weight.size()[1]):
                if np.sum(np.absolute(weight[:, i, :, :].cpu().detach().numpy())) == 0:
                    empty_channel_index.append(i)

            sc_empty_channels[layer_cont] = empty_channel_index

            # =====================================================================================#
            """ calculate unused channel, empty filter by filter pruning => next layer channel """
            empty_filter_index = []

            for j in range(weight.size()[0]):
                if np.sum(np.absolute(weight[j, :, :, :].cpu().detach().numpy())) == 0:
                    empty_filter_index.append(j)

            sc_empty_filters[layer_cont] = empty_filter_index

        layer_cont += 1  # count for each layer

    return conv_empty_channels, conv_empty_filters, sc_empty_channels, sc_empty_filters


def remove_unused_path_resnet18(model, conv_empty_channels, conv_empty_filters, sc_empty_channels, sc_empty_filters):
    # channel_to_remove, filter_to_remove,
    # sc_empty_channels, sc_empty_filters,
    # conv_layer_number, shortcut_layer_number

    layer_cont = 1
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):
            # ================= layer 1 ==============================
            if layer_cont == 1:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                # no channel remove in first layer
                # ------------ remove filter ----------------
                for index in conv_empty_channels[4]:
                    if index in conv_empty_channels[10] and index in conv_empty_channels[16] and index in sc_empty_channels[22]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 4 ==============================
            if layer_cont == 4:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[1]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[7]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 7 ==============================
            if layer_cont == 7:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[4]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[10]:
                    if index in conv_empty_channels[16] and index in sc_empty_channels[22]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 10 ==============================
            if layer_cont == 10:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[7]:
                    if index in conv_empty_filters[1]:
                        #print("index {} is in sc".format(index))
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[13]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 13 ==============================
            if layer_cont == 13:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[10]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[16]:
                    if index in sc_empty_channels[22]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 16 ==============================
            if layer_cont == 16:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[13]:
                    if index in conv_empty_filters[7]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[19]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 19 ==============================
            if layer_cont == 19:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[16]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[25]:
                    if index in sc_empty_filters[22] and index in sc_empty_channels[37] and index in conv_empty_channels[31]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= shortcut layer 22 ==============================
            if layer_cont == 22:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[13]:
                    if index in conv_empty_filters[7] and index in conv_empty_filters[1]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[25]:
                    if index in conv_empty_channels[31] and index in sc_empty_channels[37]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 25 ==============================
            if layer_cont == 25:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[19]:
                    if index in sc_empty_filters[22]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[28]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 28 ==============================
            if layer_cont == 28:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[25]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[31]:
                    if index in sc_empty_channels[37]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 31 ==============================
            if layer_cont == 31:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[28]:
                    if index in conv_empty_filters[19] and index in sc_empty_filters[22]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[34]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 34 ==============================
            if layer_cont == 34:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[31]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[40]:
                    if index in conv_empty_channels[46] and index in sc_empty_channels[52]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= shortcut layer 37 ==============================
            if layer_cont == 37:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[28]:
                    if index in conv_empty_filters[19] and index in sc_empty_filters[22]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[40]:
                    if index in conv_empty_channels[46] and index in sc_empty_channels[52]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 40 ==============================
            if layer_cont == 40:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[34]:
                    if index in sc_empty_filters[37]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[43]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 43 ==============================
            if layer_cont == 43:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[40]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[46]:
                    if index in sc_empty_channels[52]:
                        newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 46 ==============================
            if layer_cont == 46:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[43]:
                    if index in conv_empty_filters[34] and index in sc_empty_filters[37]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[49]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 49 ==============================
            if layer_cont == 49:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[46]:
                    newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
                " may not work."
                " did't take shortcut into account (because fc layer) !!!!!!!!!! "
                # for index in conv_empty_channels[55]:
                #     newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= shortcut layer 52 ==============================
            if layer_cont == 52:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[43]:
                    if index in conv_empty_filters[34] and index in sc_empty_filters[37]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
                " may not work."
                " did't take shortcut into account (because fc layer) !!!!!!!!!! "
                # for index in conv_empty_channels[55]:
                #     newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 55 ==============================
            if layer_cont == 55:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                for index in conv_empty_filters[49]:
                    if index in sc_empty_filters[52]:
                        newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                for index in conv_empty_channels[58]:
                    newWeight[index, :, :, :] = 0  # set channel to 0

                weight.data = torch.from_numpy(newWeight).cuda()

            # ================= layer 58 ==============================
            " from test results, this layer may affect the accuracy, may because BN layer"
            if layer_cont == 58:
                newWeight = weight.cpu().detach().numpy()
                # ------------ remove channel ----------------
                # for index in conv_empty_filters[55]:
                #     newWeight[:, index, :, :] = 0  # set channel to 0
                # ------------ remove filter ----------------
                " no filter to remove in last layer"

                weight.data = torch.from_numpy(newWeight).cuda()

        layer_cont += 1

def main():
    ####################################
    ##  yolo training setting
    ####################################

    # Configure run
    data_cfg = 'cfg/coco.data'
    # train_path = parse_data_cfg(data_cfg)['train']
    # test_path = parse_data_cfg(data_cfg)['valid']
    multi_scale = False
    img_size = 416
    cfg = 'cfg/yolov3.cfg'
    lr0 = 0.001
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')

    # Initialize model
    model = Darknet(cfg, img_size)

    # model.load_state_dict(torch.load("./weights/yolov3_retrained_acc_0.489_3rhos_config_yolov3_filter.pt"))

    model.load_state_dict(torch.load("/home/hongjia/yolov3_retrained_acc_0.346_4rhos_config_yolov3_v1_filter.pt"))
    #
    # for name, weight in model.named_parameters():
    #     if (len(weight.size()) == 1):
    #         print(name)
    #         print(weight)
    # #
    # conv_empty_channels, conv_empty_filters =  find_empty_channel_and_filters_yolo(model)

    # print(conv_empty_filters[0])
    # print(conv_empty_filters[1])
    # print(conv_empty_filters[2])
    # print(conv_empty_filters[3])

    # test(model, conv_empty_channels, conv_empty_filters)

    # remove_filter, remove_channel = remove_unused_path_yolo(model, conv_empty_channels, conv_empty_filters)
    # print(remove_filter)
    # print(remove_channel)

    #
    # model.load_state_dict(torch.load("./weights/yolov3_remove.pt"))
    test_filter_sparsity(model)
    test_column_sparsity(model)

if __name__ == '__main__':
    main()