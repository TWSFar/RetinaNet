from apex import amp
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda:0')


class FPN(nn.Module):
    def __init__(self, out_channel=10):
        super(FPN, self).__init__()
        self.conv = nn.Conv2d(3, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


model = FPN().to(device)
optimizer = optim.Adam(model.parameters())


model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = torch.nn.DataParallel(model, [0, 1])
for i in range(10):
    inputs = torch.rand(2, 3, 10, 10).to(device)
    labels = torch.rand(2, 10, 10, 10).to(device)
    criterion = nn.MSELoss().to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')

# Restore
model = FPN().to(device)
optimizer = optim.Adam(model.parameters())
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])
for i in range(10):
    inputs = torch.rand(2, 3, 10, 10).to(device)
    labels = torch.rand(2, 10, 10, 10).to(device)
    criterion = nn.MSELoss().to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()