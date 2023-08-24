import torch

class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, batch_size):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = torch.nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # Initializer
            torch.nn.init.normal(deconv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

        # Residual layer
        self.residual_layer = torch.nn.Sequential()
        res = torch.nn.Conv2d(in_channels=batch_size,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.residual_layer.add_module('res', res)
        torch.nn.init.normal(res.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(res.bias, 0.0)

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        out_ = out.permute(1,0,2,3)
        adv_out = self.residual_layer(out_)
        adv_out_ = adv_out.permute(1,0,2,3)
        return adv_out_
