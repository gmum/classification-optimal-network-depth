from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn

from utils import get_device


def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Parameter):
        m.fill_(1.0) / m.size(0)
    else:
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def kaiming_init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm2d):
        # m.weight.data.normal_(1.0, 0.02)
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Parameter):
        m.fill_(1.0) / m.size(0)
    else:
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


class FCNet(nn.Module):

    def __init__(self, image_size, channels, num_layers, layer_size, classes, beta=0, gamma=False, baseline=False):
        super().__init__()
        assert image_size > 1
        assert channels >= 1
        assert num_layers > 1
        assert layer_size > 1
        assert classes > 1
        self.image_size = image_size
        self.channels = channels
        self.num_heads = num_layers
        self.layer_size = layer_size
        self.classes = classes
        self.beta = beta
        self.gamma = gamma
        self.baseline = baseline
        if not self.baseline:
            # weights for layers
            self.ws = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float, requires_grad=True)
            self.layer_penalties = torch.arange(0.0, self.num_heads, device=get_device()) + 1.0
        # layer lists
        self.layers = nn.ModuleList()
        if not self.baseline:
            self.heads = nn.ModuleList()
        # first layer
        self.layers.append(nn.Linear(self.image_size * self.channels, self.layer_size))
        if not self.baseline:
            self.heads.append(nn.Linear(self.layer_size, self.classes))
        num_layers -= 1
        # remaining layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.layer_size, self.layer_size))
            if not self.baseline:
                self.heads.append(nn.Linear(self.layer_size, self.classes))
        if self.baseline:
            self.layers.append(nn.Linear(self.layer_size, self.classes))

    def init_layer_importances(self, init='uniform'):
        if not self.baseline:
            if isinstance(init, str):
                if init == 'uniform':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                elif init == 'first':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[0] = 10.0
                elif init == 'last':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[-1] = 10.0
                else:
                    raise ValueError('Incorrect init type')
            else:
                self.ws.data = init

    def importance_parameters(self):
        return [self.ws]

    def forward(self, x):
        x = x.view(-1, self.image_size * self.channels)
        if not self.baseline:
            layer_outputs = []
            for i, (fc_layer, c_head) in enumerate(zip(self.layers, self.heads)):
                x = torch.relu(fc_layer(x))
                layer_outputs.append(torch.log_softmax(c_head(x), dim=1))
            layer_outputs_tensor = torch.stack(layer_outputs, dim=2)
            y_pred = torch.matmul(layer_outputs_tensor, torch.softmax(self.ws, dim=0))
            if self.gamma:
                exped_sum = y_pred.exp().sum(dim=1, keepdim=True)
                assert torch.isnan(exped_sum).sum().item() == 0
                assert (exped_sum == float('inf')).sum().item() == 0
                assert (exped_sum == float('-inf')).sum().item() == 0
                # assert (exped_sum == 0.0).sum().item() == 0  # this throws
                log_exped_sum = (exped_sum + 1e-30).log()
                assert torch.isnan(log_exped_sum).sum().item() == 0
                assert (log_exped_sum == float('inf')).sum().item() == 0
                assert (log_exped_sum == float('-inf')).sum().item() == 0
                y_pred -= self.gamma * log_exped_sum
                assert torch.isnan(y_pred).sum().item() == 0
                assert (y_pred == float('inf')).sum().item() == 0
                assert (y_pred == float('-inf')).sum().item() == 0
            return y_pred, layer_outputs
        else:
            for fc_layer in self.layers[:-1]:
                x = torch.relu(fc_layer(x))
            return torch.log_softmax(self.layers[-1](x), dim=1), []

    def calculate_loss(self, output, target, criterion):
        if not self.baseline:
            pen = self.beta * torch.sum(torch.softmax(self.ws, dim=0) * self.layer_penalties)
            return criterion(output, target) + pen, pen
        else:
            return criterion(output, target), torch.tensor(0.0)


class DCNet(nn.Module):

    def __init__(self, image_size, channels, num_layers, num_filters, kernel_size, classes, beta=0, gamma=False,
                 batchnorm=True, baseline=False):
        # TODO handle non-square images, different architectures, padding and stride if it becomes necessary
        super().__init__()
        assert image_size > 1
        assert channels >= 1
        assert classes > 1
        assert num_layers >= 1
        if baseline:
            assert not gamma
        self.image_size = image_size
        self.channels = channels
        self.num_heads = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        # self.filters_inc = filters_inc
        self.classes = classes
        self.beta = beta
        self.gamma = gamma
        self.batchnorm = batchnorm
        self.baseline = baseline
        if not self.baseline:
            # weights for layers
            self.ws = nn.Parameter(
                torch.zeros(self.num_heads, device=get_device(), dtype=torch.float, requires_grad=True))
            self.layer_penalties = torch.arange(0.0, self.num_heads, device=get_device()) + 1.0
        # layer lists
        self.layers = nn.ModuleList()
        if self.batchnorm:
            self.bn_layers = nn.ModuleList()
        if not self.baseline:
            self.heads = nn.ModuleList()
        # assume, for simplicity, that we only use 'same' padding and stride 1
        padding = (self.kernel_size - 1) // 2
        # layers
        c_in = self.channels
        c_out = self.num_filters
        for layer in range(self.num_heads):
            self.layers.append(nn.Conv2d(c_in, c_out, kernel_size=self.kernel_size, stride=1, padding=padding))
            c_in, c_out = c_out, c_out
            # c_in, c_out = c_out, c_out + self.filters_inc
            if self.batchnorm:
                self.bn_layers.append(nn.BatchNorm2d(c_out))
            if not self.baseline:
                self.heads.append(nn.Linear(c_out, self.classes))
        if self.baseline:
            self.layers.append(nn.Linear(c_out, self.classes))
            # self.convs.append(nn.Linear(c_out * self.image_size ** 2, self.classes))

    def init_layer_importances(self, init='uniform'):
        # TODO deduplicate
        if not self.baseline:
            if isinstance(init, str):
                if init == 'uniform':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                elif init == 'first':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[0] = 10.0
                elif init == 'last':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[-1] = 10.0
                else:
                    raise ValueError('Incorrect init type')
            else:
                self.ws.data = init

    def parameters(self):
        to_chain = [self.layers]
        if self.batchnorm:
            to_chain.append(self.bn_layers)
        if not self.baseline:
            to_chain.append(self.heads)
        yield from chain.from_iterable(m.parameters() for m in chain.from_iterable(to_chain))

    def importance_parameters(self):
        return [self.ws, ]

    def non_head_parameters(self):
        if self.batchnorm:
            return (list(l.parameters()) + list(l_bn.parameters()) for l, l_bn in zip(self.layers, self.bn_layers))
        else:
            return (list(l.parameters()) for l in self.layers)

    def forward(self, x):
        if not self.baseline:
            layer_outputs = []
            for i, (conv_layer, c_head) in enumerate(zip(self.layers, self.heads)):
                x = torch.relu(conv_layer(x))
                if self.batchnorm:
                    x = self.bn_layers[i](x)
                x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
                layer_outputs.append(torch.log_softmax(c_head(x_transformed), dim=1))
            layer_outputs_tensor = torch.stack(layer_outputs, dim=2)
            y_pred = torch.matmul(layer_outputs_tensor, torch.softmax(self.ws, dim=0))
            if self.gamma:
                exped_sum = y_pred.exp().sum(dim=1, keepdim=True)
                assert torch.isnan(exped_sum).sum().item() == 0
                assert (exped_sum == float('inf')).sum().item() == 0
                assert (exped_sum == float('-inf')).sum().item() == 0
                # assert (exped_sum == 0.0).sum().item() == 0  # this throws
                log_exped_sum = (exped_sum + 1e-30).log()
                assert torch.isnan(log_exped_sum).sum().item() == 0
                assert (log_exped_sum == float('inf')).sum().item() == 0
                assert (log_exped_sum == float('-inf')).sum().item() == 0
                y_pred -= self.gamma * log_exped_sum
                assert torch.isnan(y_pred).sum().item() == 0
                assert (y_pred == float('inf')).sum().item() == 0
                assert (y_pred == float('-inf')).sum().item() == 0
            return y_pred, layer_outputs
        else:
            for i, layer in enumerate(self.layers[:-1]):
                x = torch.relu(layer(x))
                if self.batchnorm:
                    x = self.bn_layers[i](x)
            x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
            last_activations = self.layers[-1](x_transformed)
            return torch.log_softmax(last_activations, dim=1), []

    def calculate_loss(self, output, target, criterion):
        if self.baseline:
            return criterion(output, target), torch.tensor(0.0)
        else:
            pen = self.beta * torch.sum(torch.softmax(self.ws, dim=0) * self.layer_penalties)
            if isinstance(output, list):
                layer_outputs = output
                layer_outputs_tensor = torch.stack(layer_outputs, dim=2)
                output = torch.matmul(layer_outputs_tensor, torch.softmax(self.ws, dim=0))
            return criterion(output, target) + pen, pen


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                  "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, beta=0.0, gamma=None, baseline=False):
        super(ResNet, self).__init__()
        self.baseline = baseline
        self.beta = beta
        self.gamma = gamma
        self.classes = num_classes
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        self.current_planes = 16
        # self.current_size = 32

        self.conv1 = nn.Conv2d(3, self.current_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.current_planes)
        if self.baseline:
            self.group1 = self._make_layer(block, self.planes[0], num_blocks=num_blocks[0], stride=self.strides[0])
            self.group2 = self._make_layer(block, self.planes[1], num_blocks=num_blocks[1], stride=self.strides[1])
            self.group3 = self._make_layer(block, self.planes[2], num_blocks=num_blocks[2], stride=self.strides[2])
            self.linear = nn.Linear(self.planes[2], num_classes)
        else:
            self.group1, self.heads1 = self._make_layer(block, self.planes[0], num_blocks=num_blocks[0],
                                                        stride=self.strides[0])
            self.group2, self.heads2 = self._make_layer(block, self.planes[1], num_blocks=num_blocks[1],
                                                        stride=self.strides[1])
            self.group3, self.heads3 = self._make_layer(block, self.planes[2], num_blocks=num_blocks[2],
                                                        stride=self.strides[2])
            self.num_heads = sum(len(h) for h in [self.heads1, self.heads2, self.heads3])
            self.ws = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float, requires_grad=True)
            self.layer_penalties = torch.arange(0.0, self.num_heads, device=get_device()) + 1.0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        if self.baseline:
            layers = []
        else:
            layers = nn.ModuleList()
            c_heads = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.current_planes, planes * block.expansion, stride))
            self.current_planes = planes * block.expansion
            # self.current_size /= stride
            if not self.baseline:
                head = nn.ModuleList([
                    # nn.Conv2d(self.current_planes, self.planes[-1], kernel_size=3, stride=1, padding=1),
                    # nn.Linear(self.planes[-1], self.classes),
                    nn.Linear(self.current_planes, self.classes),
                ])
                c_heads.append(head)
        if self.baseline:
            return nn.Sequential(*layers)
        else:
            return layers, c_heads

    def init_layer_importances(self, init='uniform'):
        if not self.baseline:
            if isinstance(init, str):
                if init == 'uniform':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                elif init == 'first':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[0] = 10.0
                elif init == 'last':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[-1] = 10.0
                else:
                    raise ValueError('Incorrect init type')
            else:
                self.ws.data = init

    def parameters(self):
        to_chain = [self.conv1, self.bn1, self.group1, self.group2, self.group3]
        if not self.baseline:
            to_chain.extend((self.heads1, self.heads2, self.heads3))
        else:
            to_chain.append(self.linear)
        yield from chain.from_iterable(m.parameters() for m in chain(to_chain))

    def importance_parameters(self):
        return [self.ws]

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        if self.baseline:
            x = self.group1(x)
            x = self.group2(x)
            x = self.group3(x)
            x = F.avg_pool2d(x, x.size(3))
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return torch.log_softmax(x, dim=1), []
        else:
            layer_outputs = []
            groups = chain(self.group1, self.group2, self.group3)
            heads = chain(self.heads1, self.heads2, self.heads3)
            for i, (resnet_layer, c_head) in enumerate(zip(groups, heads)):
                x = resnet_layer(x)
                x_head = F.avg_pool2d(x, x.size(3))
                x_head = x_head.view(x_head.size(0), -1)
                x_head = c_head[0](x_head)
                layer_outputs.append(torch.log_softmax(x_head, dim=1))
            layer_outputs_tensor = torch.stack(layer_outputs, dim=2)
            y_pred = torch.matmul(layer_outputs_tensor, torch.softmax(self.ws, dim=0))
            if self.gamma:
                exped_sum = y_pred.exp().sum(dim=1, keepdim=True)
                assert torch.isnan(exped_sum).sum().item() == 0
                assert (exped_sum == float('inf')).sum().item() == 0
                assert (exped_sum == float('-inf')).sum().item() == 0
                # assert (exped_sum == 0.0).sum().item() == 0  # this throws
                log_exped_sum = (exped_sum + 1e-30).log()
                assert torch.isnan(log_exped_sum).sum().item() == 0
                assert (log_exped_sum == float('inf')).sum().item() == 0
                assert (log_exped_sum == float('-inf')).sum().item() == 0
                y_pred -= self.gamma * log_exped_sum
                assert torch.isnan(y_pred).sum().item() == 0
                assert (y_pred == float('inf')).sum().item() == 0
                assert (y_pred == float('-inf')).sum().item() == 0
            return y_pred, layer_outputs

    def calculate_loss(self, output, target, criterion):
        if not self.baseline:
            pen = self.beta * torch.sum(torch.softmax(self.ws, dim=0) * self.layer_penalties)
            return criterion(output, target) + pen, pen
        else:
            return criterion(output, target), torch.tensor(0.0)


def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])


def ResNet110():
    return ResNet(BasicBlock, [18, 18, 18])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# TODO update
class VGG(nn.Module):
    def __init__(self, num_classes=10, beta=0.0, cfg=None, batch_norm=False, baseline=False):
        super().__init__()
        self.baseline = baseline
        self.beta = beta
        self.classes = num_classes
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        self.layers = nn.ModuleList()
        if not self.baseline:
            self.heads = nn.ModuleList()
        in_channels = 3
        layer_buf = nn.ModuleList()
        for v in cfg:
            if v == 'M':
                layer_buf.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_buf.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                if batch_norm:
                    layer_buf.append(nn.BatchNorm2d(v))
                self.layers.append(layer_buf)
                layer_buf = nn.ModuleList()
                in_channels = v
                if not self.baseline:
                    self.heads.append(nn.Linear(in_channels, num_classes))
        if not self.baseline:
            self.num_heads = len(self.heads)
            self.ws = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float, requires_grad=True)
            self.layer_penalties = torch.arange(0.0, self.num_heads, device=get_device()) + 1.0
        elif self.baseline == 'original':
            self.head = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            self.head = nn.Linear(in_channels, num_classes)

    def init_layer_importances(self, init='uniform'):
        if not self.baseline:
            if isinstance(init, str):
                if init == 'uniform':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                elif init == 'first':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[0] = 10.0
                elif init == 'last':
                    self.ws.data = torch.zeros(self.num_heads, device=get_device(), dtype=torch.float)
                    self.ws.data[-1] = 10.0
                else:
                    raise ValueError('Incorrect init type')
            else:
                self.ws.data = init

    def importance_parameters(self):
        return [self.ws]

    def forward(self, x):
        if not self.baseline:
            layer_outputs = torch.zeros((x.size(0), self.classes, self.num_heads), device=get_device())
            for i, (layer_group, head) in enumerate(zip(self.layers, self.heads)):
                for layer in layer_group:
                    x = layer(x)
                x = torch.relu(x)
                x_gmp = nn.functional.max_pool2d(x, (x.size(2), x.size(3)))
                layer_outputs[:, :, i] = torch.log_softmax(head(x_gmp.view(x_gmp.size(0), -1)), dim=1)
            return torch.matmul(layer_outputs, torch.softmax(self.ws, dim=0)), layer_outputs
        elif self.baseline == 'original':
            for layer_group in self.layers:
                for layer in layer_group:
                    x = layer(x)
                x = torch.relu(x)
            x_gmp = nn.functional.adaptive_avg_pool2d(x, (7, 7)).view(x.size(0), -1)
            return torch.log_softmax(self.head(x_gmp), dim=1), []
        else:
            for layer_group in self.layers:
                for layer in layer_group:
                    x = layer(x)
                x = torch.relu(x)
            x_gmp = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
            return torch.log_softmax(self.head(x_gmp), dim=1), []

    def calculate_loss(self, output, target, criterion):
        if not self.baseline:
            pen = self.beta * torch.sum(torch.softmax(self.ws, dim=0) * self.layer_penalties)
            return criterion(output, target) + pen, pen
        else:
            return criterion(output, target), torch.tensor(0.0)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            m.fill_(1.0) / m.size(0)
        else:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
