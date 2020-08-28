import torch
#codes = torch.load('codes.pt');
from torch.autograd import Variable

def sample(mu,std):
    mu = torch.cuda.FloatTensor(mu)
    std = torch.cuda.FloatTensor(std)
    eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
    eps = torch.cuda.FloatTensor(eps)
    f = eps.mul(std).add(mu)
    return f

def codeload(mu,std):
    codes = torch.load('codes.pt')
#    codes = codes.cpu()
    code = codes[0:16,:]     
    codes = Variable(torch.FloatTensor(codes))
    return codes
