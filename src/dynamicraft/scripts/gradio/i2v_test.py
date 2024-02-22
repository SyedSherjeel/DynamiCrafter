import os
import time
import logging

from omegaconf import OmegaConf
import torch
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling, get_latent_z
from utils.utils import instantiate_from_config

from einops import repeat
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything


class Image2Video():
    def __init__(self, model_dir: str, gpu_id=1, resolution='256_256') -> None:
        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1])) 
        
        self.transform = transforms.Compose([
                                            transforms.Resize(min(self.resolution)),
                                            transforms.CenterCrop(self.resolution),
                                            ])

        self.model_list = self.load_model(model_dir, gpu_id, resolution)

    def load_model(self, model_dir: str, gpu_id: str, resolution: str):
        start = time.perf_counter()
        
        ckpt_path = os.path.join(model_dir, 'checkpoints/dynamicrafter_'+resolution.split('_')[1]+'_v1/model.ckpt')
        config_file = os.path.join(model_dir, 'configs/inference_'+resolution.split('_')[1]+'_v1.0.yaml')
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        model_list = []
        model = instantiate_from_config(model_config)
        model = model.cuda(gpu_id)
        assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()
        model_list.append(model)
        
        logging.info(f"Model loaded in: {time.perf_counter()-start}")
        return model_list

    def __call__(self, image, prompt, savedir, fps,steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123):
        start = time.perf_counter()
        
        seed_everything(seed) 
        
        torch.cuda.empty_cache()
        
        gpu_id=0
        if steps > 60:
            steps = 60 
        model = self.model_list[gpu_id]
        model = model.cuda()
        batch_size=1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        text_emb = model.get_learned_conditioning([prompt])

        # img cond
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = self.transform(img_tensor) #3,h,w
        videos = image_tensor_resized.unsqueeze(0) # bchw
        
        z = get_latent_z(model, videos.unsqueeze(2)) #bc,1,hw
        
        img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

        cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
        img_emb = model.image_proj_model(cond_images)

        imtext_cond = torch.cat([text_emb, img_emb], dim=1)

        fs = torch.tensor([fs], dtype=torch.long, device=model.device)
        cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
        
        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
        ## b,samples,c,t,h,w
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        prompt_str=prompt_str[:40]
        if len(prompt_str) == 0:
            prompt_str = 'empty_prompt'

        save_videos(batch_samples, savedir, fps=self.save_fps)

        logging.info(f"Inference Completed In {time.perf_counter()-start}")
        
        return savedir

if __name__ == '__main__':
    i2v = Image2Video()
    video_path = i2v.get_image('prompts/art.png','man fishing in a boat at sunset')
    print('done', video_path)