import os
import random
from datetime import datetime
import json
from PIL import Image, ExifTags
import numpy as np
from deep_translator import GoogleTranslator
import torch
from comfy import model_management

import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
from .arch import *

def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename

def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


class HumanStyler:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STYLE_STRING",)
    FUNCTION = "run"
    CATEGORY = "XorbisUtils"

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'styler.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.styles = f.readlines()
                self.styles = [style for style in self.styles if style != '\n']
        except Exception as e:
            print(e)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "randomize": ("STRING", {"default": '', "multiline": False}),
            }
        }

    def run(self):
        style = random.choice(self.styles)
        return (style,)

class ConvertMonochrome():
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    CATEGORY = "XorbisUtils"
    FUNCTION = "black_or_white_fast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required" : {
                "image": ("IMAGE", ),
                "threshold" : ("INT", {"default": 200, "min": 0, "max": 255}),
            }
        }

    def black_or_white_fast(self, image, threshold):
        """
        threshold : higher -> flexible
        """
        # Convert image to numpy array
        data = np.array(image)

        is_nearly_black = (data < threshold).all(axis=-1)
        data[is_nearly_black] = [0, 0, 0]
        data[~is_nearly_black] = [255, 255, 255]

        # Convert array back to PIL Image
        return (torch.Tensor(data),)


class SaveLogInfo:
    RETURN_TYPES = ()
    FUNCTION = "save_log"
    CATEGORY = "XorbisUtils"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required" : {
                "save_path": ("STRING", {"default": '', "multiline": False}),
                "info_path": ("STRING", {"default": '', "multiline": False}),
                "json_message": ("STRING", {"default": '', "multiline": False}),
                "images": ("IMAGE", ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "modelname": (folder_paths.get_filename_list("checkpoints"),),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("STRING", {"default": 'unknown', "multiline": True}),
                "negative": ("STRING", {"default": 'unknown', "multiline": True}),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
            }
        }

    def process_metadata(self, steps, cfg, modelname, sampler_name, scheduler, positive, negative, seed_value, width, height):
        basemodelname = parse_name(modelname)
        comment = f"Positive prompt: {handle_whitespace(positive)}\nNegative prompt: {handle_whitespace(negative)}\nModel: {basemodelname}\nSteps: {steps}\nSampler: {sampler_name}{f'_{scheduler}' if scheduler != 'normal' else ''}\nCFG Scale: {cfg}\nSeed: {seed_value}\nSize: {width}x{height}"
        return comment


    def save_log(self, save_path, info_path, json_message, images, steps, cfg, modelname, sampler_name, scheduler, positive, negative, seed_value, width, height):
        save_path  = folder_paths.get_annotated_filepath(save_path.strip())
        info_path  = folder_paths.get_annotated_filepath(info_path.strip())

        json_message = json.loads(json_message.replace("'", '"'))
        uid = json_message['uid']
        uid_path = os.path.join(save_path, uid)
        os.makedirs(uid_path, exist_ok=True)

        # save image
        for idx, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i,0,255).astype(np.uint8))
            file_n = f'{idx}.png'
            img.save(os.path.join(uid_path, file_n))

        # create emtpy file named uuid
        empty_file_n = f'{uid}.info'
        with open(os.path.join(save_path, empty_file_n), 'w') as f:
            pass
            f.close()

        # save info file
        info_file_n = f'{uid}.txt'

        with open(os.path.join(info_path, info_file_n), 'w', encoding='utf-8') as f:
            f.writelines(f'User Response: {json_message["prompt"]}\n')
            comment = self.process_metadata(steps, cfg, modelname, sampler_name, scheduler, positive, negative, seed_value, width,
                                  height)
            f.write(comment)
            f.close()

        return {}

class RT4KSR_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                              "scale": ("INT", {"default":3}),
                              "safe_load": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name, scale, safe_load):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = model_management.get_torch_device()

        model_path = folder_paths.get_full_path("upscale_models", model_name)
        checkpoint = torch.load(model_path, map_location=device)
        model = torch.nn.DataParallel(RT4KSR_Rep(upscale=scale)).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return (model, )

        # model_path = folder_paths.get_full_path("upscale_models", model_name)
        # checkpoint = torch.load(model_path)
        # model = RT4KSR_Rep(upscale=scale)
        # net = model.load_state_dict(checkpoint['state_dict'], strict=True)
        # return (net, )



class Upscale_RT4SR:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = model_management.get_torch_device()
        #
        # memory_required = model_management.module_size(upscale_model)
        # memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        # memory_required += image.nelement() * image.element_size()
        # model_management.free_memory(memory_required, device)
        #
        # upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        output = upscale_model(in_img)
        # upscale_model.cpu()
        s = torch.clamp(output.movedim(-3,-1), min=0, max=1.0)
        return (s,)


NODE_CLASS_MAPPINGS = {
    "Save Log Info": SaveLogInfo,
    "Add Human Styler" : HumanStyler,
    "Convert Monochrome" : ConvertMonochrome,
    "RT4KSR Loader" : RT4KSR_loader,
    "Upscale RT4SR" : Upscale_RT4SR
}

NODE_DISPLAY_NAME_MAPPINGS = { "Save log info" : "Save Log Info" }

