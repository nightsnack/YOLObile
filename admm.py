from __future__ import print_function
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import operator
from numpy import linalg as LA
import numpy as np
import yaml
import random
# from testers import *


class ADMM:
    def __init__(self, model, file_name, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        self.init(file_name, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream, Loader=yaml.FullLoader)
                self.prune_ratios = raw_dict['prune_ratios']
                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho
                for (name, W) in model.module.named_parameters() if type(
                         model) is nn.parallel.DistributedDataParallel else model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                    # if(len(W.size()) == 4):
                    #     if name not in self.prune_ratios:
                    #         continue
                    #     self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    #     self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z


            except yaml.YAMLError as exc:
                print(exc)

    def adjust_rho(self, new_rho):
        self.rho = new_rho
        for k, v in self.prune_ratios.items():
            self.rhos[k] = self.rho

def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(args, weight, prune_ratio, cross_x=4, cross_f=1):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (args.sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (args.sparsity_type == "crossbar"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        crossbar_num_f = math.ceil(shape2d[0] / cross_f)
        crossbar_num_x = math.ceil(shape2d[1] / cross_x)

        for x in range(crossbar_num_x):
            # print("x={}/{}".format(x,crossbar_num_x))
            for f in range(crossbar_num_f):
                # print("f={}/{}".format(f, crossbar_num_f))
                if x != crossbar_num_x - 1 and f != crossbar_num_f - 1:
                    frag = weight2d[f * cross_f:(f + 1) * cross_f, x * cross_x:(x + 1) * cross_x]

                elif x == crossbar_num_x - 1 and f != crossbar_num_f - 1:
                    frag = weight2d[f * cross_f:(f + 1) * cross_f, x * cross_x:shape2d[1]]

                elif x != crossbar_num_x - 1 and f == crossbar_num_f - 1:
                    frag = weight2d[f * cross_f:shape2d[0], x * cross_x:(x + 1) * cross_x]

                else:  # x == crossbar_num_x - 1 and f == crossbar_num_f - 1:
                    frag = weight2d[f * cross_f:shape2d[0], x * cross_x:shape2d[1]]

                total = np.sum(frag)
                if total >= 0:
                    index = (frag < 0)
                    frag[index] = 0
                else:
                    index = (frag > 0)
                    frag[index] = 0
                # change frag will change weight2d as well
        above_threshold = weight != 0
        above_threshold = above_threshold.astype(np.float32)
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (args.sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "column-number"):
        remain = percent/100

        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        percent = 100*(1- (remain/shape2d[1]))
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        # weight2d[weight2d < 1e-40] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "column-irregular"):
        column = args.block_column
        row = args.block_fliter
        percent = args.block_prune_ratio
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        sorted_l2_norm = np.argsort(column_l2_norm)
        remainder = 0 if shape2d[1]<=column else shape2d[1]%column

        pruned_column = sorted_l2_norm[0:remainder]
        divisible_weight2d = np.delete(weight2d, pruned_column, 1)
        divisible_shape2d = divisible_weight2d.shape
        # block_num = 1 if column >= shape2d[1]  else -1

        block_num = (divisible_shape2d[0]*divisible_shape2d[1])/(min(divisible_shape2d[0],row)* min(divisible_shape2d[1],column))
        block_num = 1 if block_num<=1 else block_num
        assert block_num%1 == 0
        divisible_weight2d = divisible_weight2d.reshape(int(block_num), min(row,shape2d[0]), min(column, shape2d[1]) )

        expand_above_threshold = []
        for index, block in enumerate(divisible_weight2d):
            block = np.abs(block)
            percentile = np.percentile(block, percent, axis=1).reshape(-1,1)
            under_threshold = block <= percentile
            above_threshold = block > percentile
            divisible_weight2d[index][under_threshold] = 0
            expand_above_threshold.append( above_threshold.astype(np.float32))
        divisible_weight2d = divisible_weight2d.reshape(divisible_shape2d)
        expand_above_threshold = np.array(expand_above_threshold).reshape(divisible_shape2d)
        for pcol in np.sort(pruned_column):
            expand_above_threshold = np.insert(expand_above_threshold, pcol, 0, axis=1)
            divisible_weight2d = np.insert(divisible_weight2d, pcol, 0, axis=1)
        weight = divisible_weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)

        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (args.sparsity_type == "column-irregular-2"):
        column = args.block_column
        row = args.block_fliter
        percent = args.block_prune_ratio
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        sorted_l2_norm = np.argsort(column_l2_norm)
        remainder = 0 if shape2d[1]<=column else shape2d[1]%column

        pruned_column = sorted_l2_norm[0:remainder]
        backup = weight2d[:,pruned_column]
        divisible_weight2d = np.delete(weight2d, pruned_column, 1)
        divisible_shape2d = divisible_weight2d.shape
        # block_num = 1 if column >= shape2d[1]  else -1

        block_num = (divisible_shape2d[0]*divisible_shape2d[1])/(min(divisible_shape2d[0],row)* min(divisible_shape2d[1],column))
        block_num = 1 if block_num<=1 else block_num
        assert block_num%1 == 0
        divisible_weight2d = divisible_weight2d.reshape(int(block_num), min(row,shape2d[0]), min(column, shape2d[1]) )

        expand_above_threshold = []
        for index, block in enumerate(divisible_weight2d):
            block = np.abs(block)
            percentile = np.percentile(block, percent, axis=1).reshape(-1,1)
            under_threshold = block <= percentile
            above_threshold = block > percentile
            divisible_weight2d[index][under_threshold] = 0
            expand_above_threshold.append( above_threshold.astype(np.float32))
        divisible_weight2d = divisible_weight2d.reshape(divisible_shape2d)
        expand_above_threshold = np.array(expand_above_threshold).reshape(divisible_shape2d)
        for pcol in np.argsort(pruned_column):
            expand_above_threshold = np.insert(expand_above_threshold, pcol, 1, axis=1)
            divisible_weight2d = np.insert(divisible_weight2d, pruned_column[pcol], weight2d[:,pruned_column[pcol]], axis=1)
        weight = divisible_weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)

        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (args.sparsity_type == "advanced-column-irregular"):
        column = args.block_column
        row = args.block_fliter
        n0column = int(percent/100)
        percent = args.block_prune_ratio
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        mask = np.zeros(shape2d, dtype=bool)
        # n0index = np.where(weight2d.any(axis=0))[0]
        column_sum = np.sum(np.abs(weight2d), axis=0)
        n0index = np.argpartition(column_sum, -n0column)[-n0column:]
        mask[:, n0index] = True
        n0weight = weight2d[:,n0index]
        n0weight = np.abs(n0weight)
        blocked_weight = np.hsplit(n0weight, n0weight.shape[1] / column)
        blocked_n0index = n0index.reshape(-1,1,column)

        for i in range(n0weight.shape[0]):
            for index, block in enumerate(blocked_weight):
                percentile = np.percentile(block[i], percent)
                conditions = np.where(block[i] <= percentile)[0]
                pruned_index = blocked_n0index[index][:, conditions]
                weight2d[i,pruned_index] = 0
                mask[i,pruned_index] = False
        weight = weight2d.reshape(shape)
        expand_above_threshold = mask.reshape(shape)

        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()


    elif (args.sparsity_type == "block-reorder"):  # xuan shen
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        # print(shape, shape2d)

        length_f = 8  # this is the block size, it could be 16 or 8
        num_channel_in_every_block = 4
        kernel_s1d = shape[2]*shape[3]
        length_x = kernel_s1d * num_channel_in_every_block  # kernel size = 3

        if shape2d[0] % length_f != 0 or shape2d[1] % length_x != 0:
            print("the layer size is not divisible")
            # return torch.from_numpy(np.array([])).cuda(), torch.from_numpy(weight).cuda()
            raise SyntaxError("block_size error")

        cross_f = int(shape2d[0] / length_f)
        cross_x = int(shape2d[1] / length_x)

        # this function will not use the reorder method
        l2_norm_record = np.zeros((cross_f, cross_x * kernel_s1d))
        for i in range(cross_f):
            for j in range(cross_x):
                block = weight2d[i * length_f: (i + 1) * length_f, j * length_x: (j + 1) * length_x]
                block_l2_norm = LA.norm(block, 2, axis=0)
                for k in range(kernel_s1d):
                    for c in range(num_channel_in_every_block):
                        l2_norm_record[i, j * kernel_s1d + k] += block_l2_norm[k + c * kernel_s1d]  # there are 4 channels in every block

        percentile = np.percentile(l2_norm_record, percent)
        # under_threshold = l2_norm_record <= percentile
        above_threshold = l2_norm_record > percentile

        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        temp_mat_inexpand_0 = np.zeros(length_f)
        temp_mat_inexpand_1 = np.ones(length_f)

        for i in range(cross_f):
            for j in range(cross_x):
                # block = weight2d[i*length_f : (i+1)*length_f, j*length_x : (j+1)*length_x]
                for k in range(kernel_s1d):
                    if above_threshold[i, kernel_s1d * j + k]:
                        for c in range(num_channel_in_every_block):
                            expand_above_threshold[i * length_f: (i + 1) * length_f,
                            j * length_x + k + kernel_s1d * c] = temp_mat_inexpand_1
                    else:
                        for c in range(num_channel_in_every_block):
                            weight2d[i * length_f: (i + 1) * length_f, j * length_x + k + kernel_s1d * c] = temp_mat_inexpand_0

        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
        ####################################
    elif (args.sparsity_type == "pattern"):
        print("pattern pruning...", weight.shape)
        shape = weight.shape

        pattern1 = [[0, 0], [0, 2], [2, 0], [2, 2]]
        pattern2 = [[0, 0], [0, 1], [2, 1], [2, 2]]
        pattern3 = [[0, 0], [0, 1], [2, 0], [2, 1]]
        pattern4 = [[0, 0], [0, 1], [1, 0], [1, 1]]

        pattern5 = [[0, 2], [1, 0], [1, 2], [2, 0]]
        pattern6 = [[0, 0], [1, 0], [1, 2], [2, 2]]
        pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1]]
        pattern8 = [[0, 1], [0, 2], [2, 1], [2, 2]]

        pattern9 = [[1, 0], [1, 2], [2, 0], [2, 2]]
        pattern10 = [[0, 0], [0, 2], [1, 0], [1, 2]]
        pattern11 = [[1, 1], [1, 2], [2, 1], [2, 2]]
        pattern12 = [[1, 0], [1, 1], [2, 0], [2, 1]]
        pattern13 = [[0, 1], [0, 2], [1, 1], [1, 2]]

        patterns_dict = {1 : pattern1,
                         2 : pattern2,
                         3 : pattern3,
                         4 : pattern4,
                         5 : pattern5,
                         6 : pattern6,
                         7 : pattern7,
                         8 : pattern8,
                         9 : pattern9,
                         10 : pattern10,
                         11 : pattern11,
                         12 : pattern12,
                         13 : pattern13
                         }

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {} # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "random-pattern"):
        print("random_pattern pruning...", weight.shape)
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)

        pattern1 = [0, 2, 6, 8]
        pattern2 = [0, 1, 7, 8]
        pattern3 = [0, 1, 6, 7]
        pattern4 = [0, 1, 3, 4]

        pattern5 = [2, 3, 5, 6]
        pattern6 = [0, 3, 5, 8]
        pattern7 = [1, 2, 6, 7]
        pattern8 = [1, 2, 7, 8]

        pattern9 = [3, 5, 6, 8]
        pattern10 = [0, 2, 3, 5]
        pattern11 = [4, 5, 7, 8]
        pattern12 = [3, 4, 6, 7]
        pattern13 = [1 ,2 ,4, 5]

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
                         13: pattern13
                         }

        for i in range(shape[0]):
            zero_idx = []
            for j in range(shape[1]):
                pattern_j = np.array(patterns_dict[random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])])
                zero_idx.append(pattern_j + 9 * j)
            zero_idx = np.array(zero_idx)
            zero_idx = zero_idx.reshape(1, -1)
            # print(zero_idx)
            weight2d[i][zero_idx] = 0

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")


def hard_prune(args, ADMM, model, option=None, cross_x=4, cross_f=1):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    print("hard pruning")
    for (name, W) in model.module.named_parameters() if type(
                         model) is nn.parallel.DistributedDataParallel else model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None:
            _, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_ratios[name], cross_x, cross_f)  # get sparse model in cuda

        elif option == "random":
            _, cuda_pruned_weights = random_pruning(args, W, ADMM.prune_ratios[name])

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(args, W, ADMM.prune_ratios[name])
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable



def admm_initialization(args, ADMM, model, cross_x=4, cross_f=1):
    if not args.admm:
        return
    for i, (name, W) in enumerate(model.module.named_parameters() if type(
                         model) is nn.parallel.DistributedDataParallel else model.named_parameters()):
        if name in ADMM.prune_ratios:
            _, updated_Z = weight_pruning(args, W, ADMM.prune_ratios[name], cross_x, cross_f)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer, cross_x=4, cross_f=1):
    if not args.admm:
        return

    if epoch != 1 and (epoch - 1) % args.admm_epochs == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.module.named_parameters() if type(
                         model) is nn.parallel.DistributedDataParallel else model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            if (args. verbose):
                Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
            ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name], ADMM.prune_ratios[name], cross_x, cross_f)  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            if (args.verbose):
                if writer:
                    writer.add_scalar('layer_{}_Wk1-Zk1'.format(name), torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item(), epoch)
                    writer.add_scalar('layer_{}_Zk1-Zk'.format(name), torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev) ** 2)).item(), epoch)

                # print ("at layer {}. W(k+1)-Z(k+1): {}".format(name,torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item()))
                # print ("at layer {}, Z(k+1)-Z(k): {}".format(name,torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item()))
            ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)

def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm:

        for i, (name, W) in enumerate(model.module.named_parameters() if type(
                         model) is nn.parallel.DistributedDataParallel else model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_ratios:
                continue

            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = args.admm_epochs
    lr = None
    if epoch % admm_epoch == 0:
        lr = args.lr
    else:
        admm_epoch_offset = epoch % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.lr * (0.1 ** (admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)