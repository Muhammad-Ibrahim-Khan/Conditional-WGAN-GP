"""
Training of WGAN network with Gradient Penalty
"""

# Imports
from model import *
from controller import *
from train_utils import gradient_penalty
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(datatype='MNIST', dataset_location=None):
    dataset_transforms = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            )
        ]
    )

    if datatype == 'MNIST':

        dataset = datasets.MNIST(root="dataset/", train=True, transform=dataset_transforms, download=True)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    else:
        dataset = datasets.ImageFolder(root=dataset_location, transform=dataset_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(NOISE_CHANNELS, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
    critic = Critic(CHANNELS_IMG, FEATURES_CRIT, NUM_CLASSES, IMG_SIZE).to(device)

    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # For Tensorboard
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, labels) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]
            labels = labels.to(device)

            for _ in range(CRTIC_ITERATIONS):
                noise = torch.randn((cur_batch_size, NOISE_CHANNELS, 1, 1)).to(device)  # Noise
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: Minimize -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                     Loss c: {loss_critic:.4f}, loss G: {loss_gen:.4f}"

                )

                with torch.no_grad():
                    fake = gen(noise, labels)

                    # Take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_scalar("Critic Loss", loss_critic, global_step=step)
                    writer_fake.add_scalar("Generator Loss", loss_gen, global_step=step)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1

