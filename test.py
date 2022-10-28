import torch
from utils.metric import SegmentationMetric

def get_test_func(args):
    if args.dataset == 'citys':
        test_func = test_citys
    else:
        test_func = test_image_class
    return test_func

class train_acc(object):
    def __init__(self, args):
        self.args = args
        if self.args.train_acc:
            if self.args.dataset == 'citys':
                metric = SegmentationMetric(args.n_classes)
            else:
                self.reset()

    @torch.autograd.no_grad()
    def reset(self):
        self.correct = 0
        self.total = 0

    @torch.autograd.no_grad()
    def update(self, i, scores, y):
        if self.args.train_acc:
            if self.args.dataset == 'citys':
                metric.update(scores, y.numpy())
            else:
                _, pred = torch.max(scores, 1)
                self.correct += torch.eq(pred, y).float().sum()
                self.total += float(y.nelement())

    @torch.autograd.no_grad()
    def return_acc(self):
        if not self.args.train_acc:
            return 0.0, 0.0
        if self.args.dataset == 'citys':
            acc1, acc2 = metric.get()
            print('Train acc1: {acc_1:.4} | Train acc2: {acc_2:.4f}'.format(acc_1=acc1, acc_2=acc2))
        else:
            if self.total > 0:
                acc1 = self.correct/self.total * 100
                self.reset()
                print('Train acc1: {acc_1:.4} | Train acc2: {acc_2:.4f}'.format(acc_1=acc1, acc_2=0.0))
                return float(acc1), 0.0
            else:
                return 0.0, 0.0


@torch.autograd.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].reshape(-1).float().sum(0) / out.size(0)
    return 100. * acc

@torch.autograd.no_grad()
def test_citys(model, loader, epoch, args):
    metric = SegmentationMetric(args.n_classes)
    metric.reset()
    model.eval()
    for i, (image, target) in enumerate(loader):
        image = image.to(args.device)
        outputs = model(image)
        pred = torch.argmax(outputs[0], 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
        pixAcc, mIoU = metric.get()
    print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (epoch, pixAcc * 100, mIoU * 100))
    return mIoU, pixAcc

@torch.autograd.no_grad()
def test_image_class(model, loader, epoch, args):
    model.eval()
    acc5 = 0
    correct = 0
    total = 0
    for i, (x, y) in enumerate(loader):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)
        scores = model(x)
        _, pred = torch.max(scores, 1)
        correct += torch.eq(pred, y).float().sum()
        total += float(y.nelement())
    acc = correct/total * 100
    print('Epoch: [{0}] \t'
          'Val acc {acc_:.3f}% \t'
          .format(int(epoch),
                  acc_=float(acc)))
    return acc, acc5
