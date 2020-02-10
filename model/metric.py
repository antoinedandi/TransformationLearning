import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def r2(output, target):
    with torch.no_grad():
        y_bar = torch.mean(output,dim=0)
        SSE = torch.sum(torch.pow(target - y_bar,2.))
        SST = torch.sum(torch.pow(output - y_bar,2.))
        return torch.div(SSE,SST)






