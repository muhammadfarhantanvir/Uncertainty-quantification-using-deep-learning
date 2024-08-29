import torch.nn as nn
import torch.nn.functional as F
from src.net_utils import check_param_list_or_scalar

class SeluNet(nn.Module):
    """ConvNet -> Max_Pool -> SELU -> ConvNet -> Max_Pool -> SELU -> FC -> SELU -> FC -> SELU -> FC -> SOFTMAX"""
    def __init__(self, channels_in:int=3, num_classes=20, conv_out_shape:int=5):
        super(SeluNet, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(conv_out_shape*conv_out_shape*16, 120)  # 5*5*16 = 400 for CIFAR (32x32)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.forward_features = False

    @property
    def name(self) -> str:
        return "SeluNet"

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.selu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        if self.forward_features: f = x.clone()
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)

        if self.forward_features:
            return x, f
    
        return x
    

class ConvNet(nn.Module):
    def __init__(self, channels_in:int=3, num_channels_out:list|int=16, kernel_size:list|int=5, conv_stride:list|int=1, max_pool:list|int=2, pool_stride:list|int=2, num_conv:int=2, num_hidden:list|int=120, num_linear:int=3, num_classes:int=10, activation=nn.ReLU):
        super(ConvNet, self).__init__()

        num_channels_out = check_param_list_or_scalar(num_channels_out, num_conv)
        kernel_size = check_param_list_or_scalar(kernel_size, num_conv)
        conv_stride = check_param_list_or_scalar(conv_stride, num_conv)
        max_pool = check_param_list_or_scalar(max_pool, num_conv)
        pool_stride = check_param_list_or_scalar(pool_stride, num_conv)
        num_hidden = check_param_list_or_scalar(num_hidden, num_linear-1)

        _feat_layers = []
        
        for i in range(num_conv):
            _channel_out = num_channels_out[i] if isinstance(num_channels_out, list) else num_channels_out
            _kernel = kernel_size[i] if isinstance(kernel_size, list) else kernel_size
            _stride = conv_stride[i] if isinstance(conv_stride, list) else conv_stride
            _pool = max_pool[i] if isinstance(max_pool, list) else max_pool
            _pool_stride = pool_stride[i] if isinstance(pool_stride, list) else pool_stride
            if i == 0:
                _channel_in = channels_in 
            elif isinstance(num_channels_out, list): 
                _channel_in = num_channels_out[i-1]
            else:
                _channel_in = num_channels_out
            
            _feat_layers.append(nn.Conv2d(_channel_in, _channel_out, _kernel, _stride))
            _feat_layers.append(activation())
            _feat_layers.append(nn.MaxPool2d(_pool, _pool_stride))
        
        _feat_layers.append(nn.Flatten())

        _class_layers = []
        
        _channel_out = num_channels_out[-1] if isinstance(num_channels_out, list) else num_channels_out
        _kernel = kernel_size[i] if isinstance(kernel_size, list) else kernel_size
        for i in range(num_linear-1):
            _hidden_out = num_hidden[i] if isinstance(num_hidden, list) else num_hidden
            if i == 0:
                _in_feat = _kernel * _kernel * _channel_out
            elif isinstance(num_hidden, list): 
                _in_feat = num_hidden[i-1]
            else:
                _in_feat = num_hidden
            
            _class_layers.append(nn.Linear(_in_feat, _hidden_out))
            _class_layers.append(activation())

        # add the final classification layer
        if isinstance(num_hidden, list):
            _in_feat = num_hidden[-1]
        elif num_hidden == 0:
            _in_feat = _kernel * _kernel * _channel_out
        else:
            num_hidden
        _class_layers.append(nn.Linear(_in_feat, num_classes))

        self.feature_enc = nn.Sequential(*_feat_layers)
        self.classifyer = nn.Sequential(*_class_layers)

        self.forward_features = False
        self.activation_str = activation.__name__
        self.num_cov = num_conv
        self.num_linear = num_linear

    @property
    def name(self) -> str:
        return f"ConvNet_{self.activation_str}_{self.num_cov}_{self.num_linear}"
    
    def forward(self, x):
        x = self.feature_enc(x)

        if self.forward_features: f = x.clone()
        x = self.classifyer(x)

        if self.forward_features:
            return x, f
    
        return x


class SeluNetV0(nn.Module):
    """ConvNet -> Max_Pool -> SELU -> ConvNet -> Max_Pool -> SELU -> FC -> SELU -> FC -> SELU -> FC -> SOFTMAX"""
    def __init__(self, num_classes=10, conv_out_shape:int=16):
        super(SeluNetV0, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1)
        # self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(conv_out_shape*conv_out_shape*6, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.forward_features = False

    @property
    def name(self) -> str:
        return "SeluNetV0"

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # x = F.selu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        if self.forward_features: f = x.clone()
        x = F.selu(self.fc1(x))
        # x = F.selu(self.fc2(x))
        x = self.fc3(x)

        if self.forward_features:
            return x, f
    
        return x
    

class SeluNetUno(nn.Module):
    """ConvNet -> Max_Pool -> SELU -> ConvNet -> Max_Pool -> SELU -> FC -> SELU -> FC -> SELU -> FC -> SOFTMAX"""
    def __init__(self, channels_in:int=3, num_classes=10, conv_out_shape:int=16):
        super(SeluNetUno, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, 6, 5, 1)
        self.fc1 = nn.Linear(conv_out_shape*conv_out_shape*6, num_classes)
        self.forward_features = False

    @property
    def name(self) -> str:
        return "SeluNetUno"

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        if self.forward_features: f = x.clone()
        x = self.fc1(x)

        if self.forward_features:
            return x, f
    
        return x
    

class SeluLinear(nn.Module):
    """
    Creates a fully-connected network with `num_layers` many linear layers in the 
    feature extractor, followed by one linear layer as the classifier.

    Uses AlphaDropout when dropout_p is larger than zero and activation is SELU.
    Caution: AlphaDropout may not be compatible with AttributeLoss, do not use together!"""
    def __init__(self, num_in:int=2, num_hidden:list|int=20, num_layers:int=3, num_classes:int=2, dropout_p:float=0.2, activation=nn.ReLU):
        super(SeluLinear, self).__init__()
        
        if isinstance(num_hidden, list):
            if len(num_hidden) == 1:
                num_hidden = num_hidden[0]
            elif len(num_hidden) != num_layers:
                print(f"length of list `num_hidden` must match `num_layers`! Using num_hidden = {num_hidden[0]}...")
                num_hidden = num_hidden[0]

        num_hidden = check_param_list_or_scalar(num_hidden, num_layers)

        _layers = []
        _dropout = nn.AlphaDropout if activation == nn.SELU else nn.Dropout 
        
        for i in range(num_layers):
            _hidden_out = num_hidden[i] if isinstance(num_hidden, list) else num_hidden
            if i == 0:
                _in_feat = num_in 
            elif isinstance(num_hidden, list): 
                _in_feat = num_hidden[i-1]
            else:
                _in_feat = num_hidden
            
            _layers.append(nn.Linear(_in_feat, _hidden_out))
            _layers.append(activation())
            if dropout_p > 0.0:
                _layers.append(_dropout(dropout_p))

        _in_feat = num_hidden[-1] if isinstance(num_hidden, list) else num_hidden

        self.feature_enc = nn.Sequential(*_layers)
        self.classifyer = nn.Linear(_in_feat, num_classes)

        self.forward_features = False

    @property
    def name(self) -> str:
        return "SeluLinear"
    
    def forward(self, x):
        x = self.feature_enc(x)

        if self.forward_features: f = x.clone()
        x = self.classifyer(x)

        if self.forward_features:
            return x, f
    
        return x
