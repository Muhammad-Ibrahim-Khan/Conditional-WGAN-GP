"""
Critic and Generator implementation from WGAN paper
Using Conditional GAN
"""

# Imports
import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Input: Number of examples(batch) x Input image channels x image width x image height
    Output: Labels(In case of MNIST digits) for the model to output.
    """
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Critic, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img + 1,
                features_d,
                kernel_size = (4, 4),
                stride = (2, 2),
                padding = 1
            ),  # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4 x 4
            nn.Conv2d(
                features_d * 8,
                1,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=0
            ),  # 1 x 1
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),  # LayerNorm <--->
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)  # N x C x img_size(H) x img_size(W)
        return self.disc(x)


class Generator(nn.Module):
    """
   Input: Number of examples(batch)
   Number of Input noise channels
   Number of Input image channels
   Number of Features for generator
   Output: Labels(In case of MNIST digits) for the model to learn and output.
   """
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            self._block(channels_noise + embed_size, features_g * 16, 4, 1, 0),  # N x f_g * 16 x 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32 x 32
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size = (4, 4),
                stride= (2, 2),
                padding= (1, 1)
            ),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, labels):
        # Latent Vector/Noise z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

