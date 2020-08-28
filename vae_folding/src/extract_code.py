import torch
a = torch.load('code&samplerec.pt')
codes = a[:,0:512]
print(codes.size())
torch.save(codes,'codes.pt')

