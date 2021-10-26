import argparse
import random
import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataset import DatasetFromHdf5

from model import dense_cbam

from utils import *
import pandas as pd

import h5py
# Training settings
parser = argparse.ArgumentParser(description="Pytorch remoteCT classification")
parser.add_argument("--batchsize", type=int, default=32, help="Training batch size")
parser.add_argument("--num_iter_toprint", type=int, default=30, help="Training patch size")
parser.add_argument("--patchsize", type=int, default=512, help="Training patch size")
parser.add_argument("--path_data_tr", default="./data/H5/ct_2D3D_allinfo.h5", type=str, help="Training datapath")#SynthesizedfromN18_256s64
parser.add_argument("--path_data_te", default="", type=str, help="Training datapath")#SynthesizedfromN18_256s64
parser.add_argument("--nEpochs", type=int, default=75, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_reduce", type=float, default=0.5, help="rate of reduction of learning rate, Default=0.4")

parser.add_argument("--augment", type=int, default=1, help="if augmentation?1 yes, 0 no")

parser.add_argument("--num_out", type=int, default=2, help="how many classes in outputs?")

parser.add_argument("--block_config", type=int, default=(8,12,8,8), help="Training patch size")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", type=str, default='0')

parser.add_argument("--resume", default="./model/dense_cbam_cmv_BloodOrCSF_onlyPIH_ct_2D3D_32_fold5of5/model_epoch_40000.pth" , type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start_epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model_files, Default=None')
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="", type=str, help='ID for training')
parser.add_argument("--model", default="dense_cbam", type=str, help="choices: densenet/raresnet/resnet")
parser.add_argument("--alter", action="store_true", help="alternation for training??")

def main():
    global opt, model, CE_tr_epoch
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.cuda
    print(opt)

    opt.cuda = True
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    cudnn.benchmark = True
    CEloss = nn.CrossEntropyLoss().cuda()

    print("===> Building model")
    model = dense_cbam(block_config=opt.block_config, num_classes=opt.num_out)
    model = torch.nn.DataParallel(model).cuda()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"===> loading checkpoint: {opt.resume}")
            checkpoint = torch.load(opt.resume)
            #opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("===> no checkpoint found at {}".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ####################################################################################################################
    # path for saving model_files of this fold
    save_path = os.path.join('.', "checkpoints", "{}".format(opt.ID))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_set = DatasetFromHdf5(opt.path_data_tr, augmentation=opt.augment)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize,
                                          shuffle=True)
    # load validation data
    CE_tr_epoch = [0.7]
    acc_epoch, sen_epoch, spe_epoch, error_epoch, acc_sen_epoch = [], [], [], [], []
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        acc, sen, spe, error  = eval(model, opt)
        acc_epoch.append(acc)
        sen_epoch.append(sen)
        spe_epoch.append(spe)
        error_epoch.append(error)
        # save best model
        max_acc, max_sen, max_spe = save_bestepoch(model, save_path, acc_epoch, sen_epoch, spe_epoch, error_epoch, epoch)
        # train model
        train(training_data_loader, optimizer, model, epoch, CEloss, opt)


def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return lr


def eval(model, opt):
    model.eval()
    ######################################################################################
    h5file = h5py.File(opt.path_data_te, 'r')
    data = h5file['data']
    GT = h5file['label']
    num_val = data.shape[0]
    out_score = np.zeros([num_val, opt.num_out])

    for i in range(num_val):
        stack = data[i:i+1] # shape : 1 x D x H x W
        stack = np.rollaxis(stack, 1, 0) # shape: D x 1 x H x W
        stack = torch.tensor(stack).cuda()
        with torch.no_grad():
            out_p = model(stack)
        out_p = out_p.cpu().numpy() # shape: D x C
        out_score[i] = np.mean(out_p, 0)

    prediction = np.argmax(out_score, 1)
    results = (prediction == GT)
    prob = np.exp(out_score[:, 1]) / (np.exp(out_score[:, 0]) + np.exp(out_score[:, 1]))
    error = np.mean(- GT * np.log(prob) - (1 - GT) * np.log(1 - prob))

    accuracy = np.mean(results)

    sensitivity = np.sum(((results == True) & (GT == 1))) / np.sum(GT == 1)
    specificity = np.sum(((results == True) & (GT == 0))) / np.sum(GT == 0)
    return  accuracy, sensitivity, specificity, error


def sensitivity_specificify(correct, labels):
    sensitivity = torch.sum((correct == 1) & (labels == 1)).float()/torch.sum(labels == 1).float()
    specificity = torch.sum((correct == 1) & (labels == 0)).float()/torch.sum(labels == 0).float()
    return sensitivity, specificity


def train(training_data_loader, optimizer, model, epoch,  CEloss, opt):
    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_tr = []
    for iteration, batch in enumerate(training_data_loader, 1):
        input_data, label = batch
        # form the tables of label and select data to train
        # input_data: N x 1 x H x W;
        # out: N x num_classes;
        # out is unnormalized
        out = model(input_data.cuda())
        CE = CEloss(out, label.view(-1).cuda())
        loss = CE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tr.append(CE.cpu().detach().numpy())
        if iteration%opt.num_iter_toprint == 0:
             print("===> Epoch[{}]({}/{},lr:{:.8f}): CE:{:.6f}".format(epoch, iteration,
                                                    len(training_data_loader), lr, CE))
    CE_tr_epoch.append(np.mean(loss_tr))


def save_bestepoch(model, save_path, acc_epoch, sen_epoch, spe_epoch, CE_epoch, epoch):
    print("===> Accuracy/Sensitivity/Specificify of epoch-{} ".format(epoch - 1))
    print("{:.4f}/{:.4f}/{:.4f}".format(acc_epoch[-1], sen_epoch[-1], spe_epoch[-1]))
    print("===>Error CE:{:.4f}".format(CE_epoch[-1]))
    max_acc = np.max(acc_epoch)
    acc_best_epoch = np.where(np.array(acc_epoch) == max_acc)[0][-1]  # find the final epoch yielding highest accuracy
    max_sen = sen_epoch[acc_best_epoch]
    max_spe = spe_epoch[acc_best_epoch]
    print("===> Highest Accuracy/Sen/Spe:ep-{}-{:4f}/{:4f}/{:4f}".format(acc_best_epoch, max_acc, sen_epoch[acc_best_epoch], spe_epoch[acc_best_epoch]))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(acc_epoch, color='tab:red')
    plt.legend(['Val Accuracy'])
    min_ce = np.min(CE_epoch)
    min_ce_epoch = np.argmin(CE_epoch)
    plt.scatter(acc_best_epoch, max_acc, color='tab:red')
    plt.scatter(min_ce_epoch, acc_epoch[min_ce_epoch], color='tab:red')
    plt.annotate('ep {}\nAcc:{:.4f}'.format(acc_best_epoch, max_acc), (acc_best_epoch + 1, max_acc))
    plt.annotate('ep {}\nAcc:{:.4f}'.format(min_ce_epoch, acc_epoch[min_ce_epoch]),
                 (min_ce_epoch + 1, acc_epoch[min_ce_epoch]))

    plt.subplot(2,1,2)
    plt.plot(CE_epoch)
    plt.plot(CE_tr_epoch)

    plt.scatter(min_ce_epoch, min_ce)
    plt.annotate('ep {}:\nCE: {:.4f}'.format(min_ce_epoch, min_ce), (min_ce_epoch + 1, min_ce))
    plt.legend(['Val Cross Entropy', 'Training Cross Entropy'])

    rn = np.random.randint(10000)
    plt.savefig(os.path.join(save_path, 'CE_acc_{}'.format(rn)), dpi=200)
    plt.close()

    df = pd.DataFrame({'accuracy': acc_epoch, 'sensitivity': sen_epoch, 'specificity': spe_epoch})
    df.to_csv(save_path+'/results_table_{}.csv'.format(rn))

    save_checkpoint(model, epoch, save_path)
    return max_acc, max_sen, max_spe


def save_checkpoint(model, epoch, save_path):
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model_files
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
