import os
import random
from datetime import datetime
import json
from PIL import Image, ExifTags
import numpy as np
from deep_translator import GoogleTranslator

import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION


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



NODE_CLASS_MAPPINGS = {
    "Save Log Info": SaveLogInfo,
    "Add Human Styler" : HumanStyler
}

NODE_DISPLAY_NAME_MAPPINGS = { "Save log info" : "Save Log Info" }

