from apex import amp
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda:0')


class FPN(nn.Module):
    def __init__(self, out_channel=10):
        super(FPN, self).__init__()
        self.conv = nn.Linear(50, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, a, b):
        a = self.conv(a)
        a = self.relu(a)
        b = b.type_as(a)
        iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
        return iw

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

model = FPN().to(device)
optimizer = optim.Adam(model.parameters())

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = torch.nn.DataParallel(model, [0, 1])
model = torch.nn.DataParallel(model, [1])
model.apply(inplace_relu)
model.eval()
with torch.no_grad():
    for i in range(10):
        inputs = torch.rand(3, 42, 50).to(device)
        labels = torch.rand(3, 10, 4).to(device)
        label2 = torch.rand(3)
        criterion = nn.MSELoss().to(device)

        try:
            outputs = model(inputs, labels)
        except:
            pass
    # loss = criterion(outputs.sum*(), labels)
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    # optimizer.step()

