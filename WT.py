# %%
# !pip install opencv-python transformers accelerate insightface
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

import pathlib
import random
num_images_per_prompt = 1

# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model = 'wangqixun/YamerMIX_v8'
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(base_model, 
                                                          controlnet=controlnet, 
                                                          torch_dtype=torch.float16)
pipe.cuda()

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)

# %%
# load an image
for original in pathlib.Path('/home/user/InstantID/Originals').iterdir():
    
    print(original)

    orignal_filename = str(original).split('/')[-1].split('.')[0]
    extension = str(original).split('/')[-1].split('.')[1]
    if extension != 'JPG':
        continue
    face_image = load_image(str(original))
    
    face_image.show()
    
    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    pipe.set_ip_adapter_scale(0.8)

    prompts = [
        'Photo of a woman skiing in Switzerland.',
        'Photo of a woman paragliding in Austria',
        'Photo of a woman trekking in Luxembourg',
        'Photo of a woman in Spain going in hot air balloon',
        'Photo of a woman rowing in Thames London',
        'Photo of a woman in a ship in Belgium',
        'Photo of a woman in a mass in Italy',
        'Photo of a woman in tulips garden at Netherlands',
        'Photo of a woman in the statue of liberty in New York'
        ]

    negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green, anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

    for prompt in prompts:
        print(prompt)

        path = pathlib.Path(f'/home/user/InstantID/Generated/{prompt}/{orignal_filename}')
        path.mkdir(parents=True, exist_ok=True)
        random.seed(9001)
        
        prompt += ' cinematic still a photo of a woman. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy'
        
        for asd in range(num_images_per_prompt):    

            random_int = random.randint(1, num_images_per_prompt)

            image = pipe(prompt=prompt, 
                        image_embeds=face_emb, 
                        image=face_kps, 
                        controlnet_conditioning_scale=0.8,
                        negative_prompt=negative_prompt,
                        random_seed=random_int,
                        ).images[0]
            
            image.show()

            image.save(f'{path}/{random_int}.jpeg')
                                                                    



# %%
