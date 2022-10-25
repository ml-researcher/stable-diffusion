from unittest.util import unorderable_list_difference
from ldm.models.diffusion.ddpm import DDPM
from omegaconf import OmegaConf

config = OmegaConf.load('configs/mnist.yaml')
params = config['model']['params']

model = DDPM.load_from_checkpoint('logs/2022-10-25T16-28-36_mnist/checkpoints/epoch=000012.ckpt', strict=False, **params)
# model = DDPM.load_from_checkpoint('logs/2022-10-25T15-04-07_mnist/checkpoints/epoch=000049.ckpt', strict=False, **params)
# model = DDPM.load_from_checkpoint('logs/2022-10-25T14-57-58_mnist/checkpoints/epoch=000049.ckpt', strict=False, **params)
model = model.cuda()
model.eval()
x = model.sample()
from ldm.data.mnist import unnormalize_to_zero_to_one
x = unnormalize_to_zero_to_one(x)
from torchvision.utils import save_image
save_image(x, 'sample.png')

# breakpoint()