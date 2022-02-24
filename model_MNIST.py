import torch.nn as nn

class SimpleLinearNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleLinearNet, self).__init__()
        self.layer0 = nn.Linear(in_channels, hidden_channels)
        self.relu0 = nn.ReLU()
        self.Htanh0 = nn.Hardtanh()
        self.layer1 = nn.Linear(hidden_channels, hidden_channels)
        self.relu1 = nn.ReLU()
        self.Htanh1 = nn.Hardtanh()
        self.layer2 = nn.Linear(hidden_channels, hidden_channels)
        self.relu2 = nn.ReLU()
        self.Htanh2 = nn.Hardtanh()
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x0 = self.layer0(x)
        x1_ = self.relu0(x0)
        # x1 = self.Htanh0(x0)
        x1 = self.layer1(x1_)
        x2_ = self.relu1(x1)
        # x2 = self.Htanh1(x1)
        x2 = self.layer2(x2_)
        x3_ = self.relu2(x2)
        # x3 = self.Htanh2(x2)
        y = self.classifier(x3_)

        return x1_, x2_, x3_, y