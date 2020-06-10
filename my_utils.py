import numpy as np
import torch
import math
import yaml
import xlsxwriter
from numpy import linalg as LA
from models import Darknet


from collections import OrderedDict

def write_to_excel(model, fc=False, bias=False, all_layer=False):

    xlse_file_name = "weights.xlsx"
    workbook = xlsxwriter.Workbook(xlse_file_name)
    print("--------------------------------------------------")
    print("writing weights to excel file {}".format(xlse_file_name))

    for name, weight in model.named_parameters():  # calculate total layer
        # not conv layer
        if not (len(weight.size()) == 4 and "shortcut" not in name and "downsample" not in name):
            # not write all_layer (bias and fc)
            if not all_layer:
                # is batch_norm or bias
                if len(weight.size()) == 1:
                    if "bias" not in name:  # is batch_norm
                        continue
                    elif not bias:  # not write bias
                        continue
                # not write fully connected layer
                elif not fc:
                    continue

        weight2d = weight.reshape(weight.shape[0], -1)  # reshape to 2d format

        worksheet = workbook.add_worksheet(name)
        print("writing {} ...".format(name))
        start_col = 0  # writing whole col from row 0
        for row in range(weight.shape[0]):
            worksheet.write_row(row, start_col, weight2d[row])

    workbook.close()

    print("writing finished!")
    print("each row contains all the weights on a filter, there are filter number of rows")


def yaml_sparsity_calculator(model, filename1=None, filename2=None):
    if filename1 == None:
        print("filename1 needs to be specified!")
        return
    # ------------- file 1 -------------
    if not isinstance(filename1, str):
        raise Exception("filename must be a str")
    file = "./prune_config/"+filename1+".yaml"
    print(file)
    with open(file, "r") as stream:
        try:
            raw_dict = yaml.full_load(stream)
            prune_ratios = raw_dict['prune_ratios']

            total_weights = 0
            remained_weishts = 0
            for name, weight in model.named_parameters():
                if(len(weight.size()) == 4 and "shortcut" not in name and "downsample" not in name):
                    weight_num = weight.size()[0] * weight.size()[1] * weight.size()[2] * weight.size()[3]
                    total_weights += weight_num
                    if name in prune_ratios:
                        remained_weishts += math.ceil(weight_num*(1-prune_ratios[name]))
                    else:
                        remained_weishts += weight_num
                        print("layer {} is not found in yaml file!".format(name))

            ratio_1 = round(total_weights/remained_weishts, 4)
            print("Conv layer prune ratio in {} is around: {}x".format(filename1, ratio_1))

        except yaml.YAMLError as exc:
            print(exc)

    # ------------- file 2 -------------
    if filename2 == None:
        return
    if not isinstance(filename2, str):
        raise Exception("filename must be a str")
    file = "./profile/" + filename2 + ".yaml"
    print(file)
    with open(file, "r") as stream:
        try:
            raw_dict = yaml.full_load(stream)
            prune_ratios = raw_dict['prune_ratios']

            total_weights = 0
            remained_weishts = 0
            for name, weight in model.named_parameters():
                if (len(weight.size()) == 4 and "shortcut" not in name and "downsample" not in name):
                    weight_num = weight.size()[0] * weight.size()[1] * weight.size()[2] * weight.size()[3]
                    total_weights += weight_num
                    if name in prune_ratios:
                        remained_weishts += math.ceil(weight_num * (1 - prune_ratios[name]))
                    else:
                        remained_weishts += weight_num
                        print("layer {} is not found in yaml file!".format(name))

            ratio_2 = round(total_weights / remained_weishts, 2)
            print("Conv layer prune ratio in {} is around: {}x".format(filename2, ratio_2))

        except yaml.YAMLError as exc:
            print(exc)

    print("The overall pruning ratio is around: {:.4f}x".format(ratio_1 * ratio_2))

def manually_hard_prune(model, yaml_name, sparsity_type, cross_x=4, cross_f=1):
    # -------- get prune ratio from yaml ---------
    if yaml_name == None:
        print("filename1 needs to be specified!")
        return
    # ------------- file 1 -------------
    if not isinstance(yaml_name, str):
        raise Exception("filename must be a str")
    file = "./prune_config/"+yaml_name+".yaml"
    print(file)
    with open(file, "r") as stream:
        try:
            raw_dict = yaml.full_load(stream)
            prune_ratios = raw_dict['prune_ratios']
        except yaml.YAMLError as exc:
            print(exc)

    print("hard pruning")
    for (name, weight) in model.named_parameters():
        if name not in prune_ratios:  # ignore layers that do not have rho
            continue

    # -------------- hard prune -----------
        weight_np = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
        percent = prune_ratios[name] * 100

        if (sparsity_type == "irregular"):
            weight_temp = np.abs(weight_np)  # a buffer that holds weights with absolute values
            percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
            under_threshold = weight_temp < percentile
            weight_np[under_threshold] = 0
            weight.data = torch.from_numpy(weight_np).cuda()

        elif (sparsity_type == "column"):
            shape = weight_np.shape
            weight2d = weight_np.reshape(shape[0], -1)
            column_l2_norm = LA.norm(weight2d, 2, axis=0)
            percentile = np.percentile(column_l2_norm, percent)
            under_threshold = column_l2_norm < percentile
            weight2d[:, under_threshold] = 0
            weight.data = torch.from_numpy(weight2d.reshape(shape)).cuda()

        elif (sparsity_type == "filter"):
            shape = weight_np.shape
            weight2d = weight_np.reshape(shape[0], -1)
            row_l2_norm = LA.norm(weight2d, 2, axis=1)
            percentile = np.percentile(row_l2_norm, percent)
            under_threshold = row_l2_norm <= percentile
            weight2d[under_threshold, :] = 0
            weight.data = torch.from_numpy(weight2d.reshape(shape)).cuda()



def test_sparsity(model, column=True, channel=True, filter=True, kernel=True):

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
                print("(empty/total) channel of {}({}) is: ({}/{}) ({}). channel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_channels, weight.size()[1], weight.size()[1]-empty_channels, empty_channels / channel_num))

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
                print("(empty/total) filter of {}({}) is: ({}/{}) ({}). filter sparsity is: {:.4f} ({:.4f})".format(
                    name, layer_cont, empty_filters, weight.size()[0], weight.size()[0]-empty_filters, empty_filters / filter_num, 1-(empty_filters / filter_num)))

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

    # --------------------- kernel sparsity --------------------
    if(kernel):

        total_kernels = 0
        total_empty_kernels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
                shape = weight.shape
                npWeight = weight.cpu().detach().numpy()
                weight3d = npWeight.reshape(shape[0], shape[1], -1)

                empty_kernels = 0
                kernel_num = weight.size()[0] * weight.size()[1]

                for i in range(weight.size()[0]):
                    for j in range(weight.size()[1]):
                        if np.sum(np.absolute(weight3d[i, j, :])) == 0:
                            empty_kernels += 1
                print("(empty/total) kernel of {}({}) is: ({}/{}) ({}). kernel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_kernels, kernel_num, kernel_num-empty_kernels, empty_kernels / kernel_num))

                total_kernels += kernel_num
                total_empty_kernels += empty_kernels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of kernels: {}, empty-kernels: {}, kernel sparsity is: {:.4f}".format(
            total_kernels, total_empty_kernels, total_empty_kernels / total_kernels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")
    return comp_ratio




if __name__ == '__main__':
    model = Darknet(cfg = 'cfg/csresnext50c-yolo-spp.cfg',img_size=(320,320))
    # state_dict = torch.load('weights/yolov3_retrained_acc_0.492_4rhos_config_yolov3_v00_column.pt') #model_prunned/yolov3_0.1_config_yolov3_v00_column.pt
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # load params
    # model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict) ##state_dict["model"]
    input = torch.randn(1, 3, 320, 320)


    yaml_sparsity_calculator(model,filename1='config_resnext50spp_v2')
    # manually_hard_prune(model,yaml_name='config_yolov3_v1',sparsity_type='column' )
    comp_ratio = test_sparsity(model, column=True, channel=False, filter=False, kernel=False)
    print(comp_ratio)
    # 320
    # flops:spp: 19659927552.0/yolov3cfg:19554916352
    # csresnext50c-spp.cfg               11591114700.0
    # params:                          62998752.0
    # csresnext50c-spp.cfg             41622845.0
    # 416
    # flops:33047797760.0
    # params:61949152.0

    # ratio:1.6809