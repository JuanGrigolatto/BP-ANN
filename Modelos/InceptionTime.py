import torch
import torch.nn as nn
from fastcore.utils import delegates

class Add(nn.Module):
    def forward(self, x, y):
        return x + y

def ifnone(a, b):
    return b if a is None else a

class InceptionModule(nn.Module):
    def __init__(self, in_channels, bottleneck=True, n_filters=32, max_kernel_size=40):
        super(InceptionModule, self).__init__()
        # Bottleneck opcional
        self.use_bottleneck = bottleneck and in_channels > 1
        bottleneck_channels = n_filters if self.use_bottleneck else in_channels
        self.bottleneck_layer = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()
        # Cálculo de kernels impares
        ks = [max_kernel_size // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, n_filters, k, bias=False) for k in ks
        ])
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1, bias=False)
        )
        self.batch_norm = nn.BatchNorm1d(n_filters * 4)
        self.activacion = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck_layer(x)
        conv_outputs = [conv(x) for conv in self.convs]
        maxpool_output = self.maxconvpool(input_tensor)
        x = torch.cat(conv_outputs + [maxpool_output], dim=1)
        x = self.batch_norm(x)
        x = self.activacion(x)
        return x
    
@delegates(InceptionModule.__init__)
class InceptionBlock(nn.module):
    def __init__(self, in_channels, n_filters=32, residual=True, depth=6, **kwargs):
        self.residular = residual
        self.depth = depth
        self.inception_modules = nn.ModuleList()
        self.shortcut= nn.ModuleList()
        for d in range(depth):
            self.inception_modules.append(InceptionModule(in_channels if d == 0 else n_filters * 4, n_filters, **kwargs))
            if self.residual and d % 3 == 2: 
                n_in, n_out = in_channels if d == 2 else n_filters * 4, n_filters * 4
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else nn.Conv1d(n_in, n_out, 1, act=None))
        self.add = Add()
        self.act = nn.ReLU()
    
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception_modules[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(self.add(x, self.shortcut[d//3](res)))
        return x
    
@delegates(InceptionModule.__init__)
class InceptionTime(nn.Module):
    def __init__(self, c_in, c_out, seq_len=None, n_filters=32, nb_filters=None, **kwargs):
        n_filters = ifnone(n_filters, nb_filters) # for compatibility
        self.inceptionblock = InceptionBlock(c_in, n_filters, **kwargs)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters * 4, c_out)

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return x