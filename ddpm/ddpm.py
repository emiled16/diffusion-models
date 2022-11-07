import torch
import torch.nn.functional as F
from models import SimpleUnet
from torch.optim import Adam


class DDPM:
    def __init__(self, betas, T=200, lr=0.001, epochs=100):
        self.betas = betas
        self.T = T
        self.epochs=epochs
        self.lr = lr

        # initialize the closed form for the process
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) \
                                    / (1. - self.alphas_cumprod)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleUnet().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def _create_noise(shape):
        return torch.randn(shape)


    def forward_diffusion(self, x_0, t):
        noise = self._create_noise(x_0.shape)
        sqrt_alphas_cumprod_t = ... #TODO
        sqrt_one_minus_alphas_cumprod_t = ... #TODO

        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
        + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), \
            noise.to(self.device)

    # TODO: 
    @torch.no_grad()
    def sample_timestep(self, x, t):
        betas_t = get_index_from_list(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def sample_plot_image(self,):
        # Sample noise
        img_size = IMG_SIZE
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(T/num_images)

        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t)
            if i % stepsize == 0:
                plt.subplot(1, num_images, i/stepsize+1)
                show_tensor_image(img.detach().cpu())
        plt.show()        

    def train(self, dataloader):
        for epoch in range(self.epochs):
            for step, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
    
                # sample time t
                t = torch.randint(0, self.T, (self.BATCH_SIZE,), device=self.device).long()
                x_noisy, noise = self.forward_diffusion_sample(batch[0], t)
                noise_pred = self.model(x_noisy, t)
                loss = F.l1_loss(noise, noise_pred)
                loss.backward()
                self.optimizer.step()
                if epoch % 5 == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    self.sample_plot_image()