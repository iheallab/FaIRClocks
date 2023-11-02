import torch.nn as nn
import torch
import copy
import torch.nn.init as init

class RF_VAE2(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data.
        Taken entirely from github.com/ThomasMrY/RF-VAE"""
    def __init__(self, z_dim=10):
        super(RF_VAE2, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def loadEncoder(rfvae_name, rfvae_dims):
    print("Loading encoder...")
    rfvae = RF_VAE2(rfvae_dims)
    checkpoint = torch.load(rfvae_name, map_location=torch.device('cpu'))
    rfvae.load_state_dict(checkpoint['model_states']['VAE'])
    # remove decoder:
    encoder = rfvae.encode
    return encoder

class NeuralNetwork(nn.Module):
    def __init__(self, encoder):
        super(NeuralNetwork, self).__init__()
        self.z_dim = 10
        self.encoder = copy.deepcopy(encoder)
        # freeze encoder layers:
        i = 0
        for layer in self.encoder:
            if i < 8:
                layer.trainable = False
            i += 1
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and '8' not in name and '10' not in name:
                param.requires_grad = False
                
        self.fully_connected = nn.Sequential(
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, 4),
            nn.ReLU(),
            nn.Linear(4, 1))
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
            
    def forward(self, x):
        stats = self.encoder(x)
        
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar).squeeze()
        logits = self.fully_connected(z)
        return logits

class NeuralNetworkDemographics(nn.Module):
    def __init__(self, encoder):
        super(NeuralNetworkDemographics, self).__init__()
        self.z_dim = 10
        self.encoder = copy.deepcopy(encoder)
        # freeze encoder layers:
        i = 0
        for layer in self.encoder:
            if i < 8:
                layer.trainable = False
            i += 1
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and '8' not in name and '10' not in name:
                param.requires_grad = False
                
        self.fully_connected = nn.Sequential(
            nn.Linear(14, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
            
    def forward(self, x, demo):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar).squeeze()
        final = torch.cat((z, demo), dim=1)
        logits = self.fully_connected(final)
        return logits

