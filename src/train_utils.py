import torch

def gradient_penalty(critic, labels, real, fake, device='cpu'):
    batch_size, channels, height, width = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate Critic Scores
    mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    # Flatten all dimensions except number of examples(batch size)
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    grad_penalty = torch.mean((gradient_norm - 1) ** 2)

    return grad_penalty

