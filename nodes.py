import cv2
import torchvision.transforms.v2 as T
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import requests
from comfy import model_management

import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
from .arch import *
from concave_hull import concave_hull_indexes
from scipy.ndimage import binary_fill_holes

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

class OneImageCompare:
    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
        file_list = [f for f in os.listdir(font_dir) if
                     os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

        return {"required": {
            "text1": ("STRING", {"multiline": True, "default": "text"}),
            "footer_height": ("INT", {"default": 100, "min": 0, "max": 1024}),
            "font_name": (file_list,),
            "font_size": ("INT", {"default": 50, "min": 0, "max": 1024}),
            "mode": (["normal", "dark"],),
            "border_thickness": ("INT", {"default": 20, "min": 0, "max": 1024}),
        },
            "optional": {
                "image1": ("IMAGE",),
            }

        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "show_help",)
    FUNCTION = "layout"

    def layout(self, text1,
               footer_height, font_name, font_size, mode,
               border_thickness, image1=None):

        from .utils import tensor2pil, apply_resize_image, text_panel, combine_images, pil2tensor

        # Get RGB values for the text and background colors
        if mode == "normal":
            font_color = "black"
            bg_color = "white"
        else:
            font_color = "white"
            bg_color = "black"

        if image1 is not None:

            img1 = tensor2pil(image1)

            # Get image width and height
            image_width, image_height = img1.width, img1.height


            # Set defaults
            margins = 50
            line_spacing = 0
            position_x = 0
            position_y = 0
            align = "center"
            rotation_angle = 0
            rotation_options = "image center"
            font_outline_thickness = 0
            font_outline_color = "black"
            align = "center"
            footer_align = "center"
            outline_thickness = border_thickness // 2
            border_thickness = border_thickness // 2

            ### Create text panel for image 1
            if footer_height > 0:
                text_panel1 = text_panel(image_width, footer_height, text1,
                                         font_name, font_size, font_color,
                                         font_outline_thickness, font_outline_color,
                                         bg_color,
                                         margins, line_spacing,
                                         position_x, position_y,
                                         align, footer_align,
                                         rotation_angle, rotation_options)

            combined_img1 = combine_images([img1, text_panel1], 'vertical')

            # Apply the outline
            if outline_thickness > 0:
                combined_img1 = ImageOps.expand(combined_img1, outline_thickness, fill=bg_color)


            # Apply the outline
            if outline_thickness > 0:
                combined_img1 = ImageOps.expand(combined_img1, outline_thickness, fill=bg_color)

            result_img = combine_images([combined_img1], 'horizontal')
        else:
            result_img = Image.new('RGB', (512, 512), bg_color)

        # Add a border to the combined image
        if border_thickness > 0:
            result_img = ImageOps.expand(result_img, border_thickness, bg_color)

        return (pil2tensor(result_img), )

    # ---------------------------------------------------------------------------------------------------------------------


class ThreeImageCompare:
    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
        file_list = [f for f in os.listdir(font_dir) if
                     os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

        return {"required": {
            "text1": ("STRING", {"multiline": True, "default": "text"}),
            "text2": ("STRING", {"multiline": True, "default": "text"}),
            "text3": ("STRING", {"multiline": True, "default": "text"}),
            "footer_height": ("INT", {"default": 100, "min": 0, "max": 1024}),
            "font_name": (file_list,),
            "font_size": ("INT", {"default": 50, "min": 0, "max": 1024}),
            "mode": (["normal", "dark"],),
            "border_thickness": ("INT", {"default": 20, "min": 0, "max": 1024}),
        },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }

        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "show_help",)
    FUNCTION = "layout"

    def layout(self, text1, text2, text3,
               footer_height, font_name, font_size, mode,
               border_thickness, image1=None, image2=None, image3=None):

        from .utils import tensor2pil, apply_resize_image, text_panel, combine_images, pil2tensor

        # Get RGB values for the text and background colors
        if mode == "normal":
            font_color = "black"
            bg_color = "white"
        else:
            font_color = "white"
            bg_color = "black"

        if image1 is not None and image2 is not None and image3 is not None:

            img1 = tensor2pil(image1)
            img2 = tensor2pil(image2)
            img3 = tensor2pil(image3)

            # Get image width and height
            image_width, image_height = img1.width, img1.height

            if img2.width != img1.width or img2.height != img1.height:
                img2 = apply_resize_image(img2, image_width, image_height, 8, "rescale", "false", 1, 256, "lanczos")

            if img3.width != img1.width or img3.height != img1.height:
                img3 = apply_resize_image(img3, image_width, image_height, 8, "rescale", "false", 1, 256, "lanczos")

            # Set defaults
            margins = 50
            line_spacing = 0
            position_x = 0
            position_y = 0
            align = "center"
            rotation_angle = 0
            rotation_options = "image center"
            font_outline_thickness = 0
            font_outline_color = "black"
            align = "center"
            footer_align = "center"
            outline_thickness = border_thickness // 2
            border_thickness = border_thickness // 2

            ### Create text panel for image 1
            if footer_height > 0:
                text_panel1 = text_panel(image_width, footer_height, text1,
                                         font_name, font_size, font_color,
                                         font_outline_thickness, font_outline_color,
                                         bg_color,
                                         margins, line_spacing,
                                         position_x, position_y,
                                         align, footer_align,
                                         rotation_angle, rotation_options)

            combined_img1 = combine_images([img1, text_panel1], 'vertical')

            # Apply the outline
            if outline_thickness > 0:
                combined_img1 = ImageOps.expand(combined_img1, outline_thickness, fill=bg_color)

            ### Create text panel for image 2
            if footer_height > 0:
                text_panel2 = text_panel(image_width, footer_height, text2,
                                         font_name, font_size, font_color,
                                         font_outline_thickness, font_outline_color,
                                         bg_color,
                                         margins, line_spacing,
                                         position_x, position_y,
                                         align, footer_align,
                                         rotation_angle, rotation_options)

            combined_img2 = combine_images([img2, text_panel2], 'vertical')

            ### Create text panel for image 3
            if footer_height > 0:
                text_panel3 = text_panel(image_width, footer_height, text3,
                                         font_name, font_size, font_color,
                                         font_outline_thickness, font_outline_color,
                                         bg_color,
                                         margins, line_spacing,
                                         position_x, position_y,
                                         align, footer_align,
                                         rotation_angle, rotation_options)

            combined_img3 = combine_images([img3, text_panel3], 'vertical')

            # Apply the outline
            if outline_thickness > 0:
                combined_img1 = ImageOps.expand(combined_img1, outline_thickness, fill=bg_color)

            if outline_thickness > 0:
                combined_img2 = ImageOps.expand(combined_img2, outline_thickness, fill=bg_color)

            if outline_thickness > 0:
                combined_img3 = ImageOps.expand(combined_img3, outline_thickness, fill=bg_color)

            result_img = combine_images([combined_img1, combined_img2, combined_img3], 'horizontal')
        else:
            result_img = Image.new('RGB', (512, 512), bg_color)

        # Add a border to the combined image
        if border_thickness > 0:
            result_img = ImageOps.expand(result_img, border_thickness, bg_color)

        return (pil2tensor(result_img), )

    # ---------------------------------------------------------------------------------------------------------------------
class MaskAlignedBbox4ConcaveHull:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image" : ("IMAGE",),
                "ori_masked_sketch" : ("MASK",),
                "masked_sketch": ("MASK",),
            },
        }
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    # MASK is for VAE Encode mask input
    # MASKED_IMAGE is for VAE Encode image input
    # MASKED_IMAGE_WITH_SKETCH is for ControlNet input
    RETURN_NAMES = ("MASK", "MASKED_IMAGE", "MASKED_IMAGE_WITH_SKETCH")
    FUNCTION = "execute"

    def fill_mask(self, mask_tensor):
        mask_np = mask_tensor.cpu().numpy()
        filled_mask_np = np.zeros_like(mask_np)

        for i in range(mask_np.shape[0]):
            mask_2d = mask_np[i]  # Extract 2D mask for single batch item
            if mask_2d.ndim > 2:  # If more than 2 dimensions, it's not a grayscale image
                raise ValueError("Mask dimension is greater than 2D which is not expected for single-batch item.")

            # Find contours in the mask
            contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw filled contours on the empty mask
            cv2.drawContours(filled_mask_np[i], contours, -1, (1,), thickness=cv2.FILLED)
        filled_mask_tensor = torch.from_numpy(filled_mask_np).to(mask_tensor.device, dtype=torch.float32)
        return filled_mask_tensor


    def execute(self, image, ori_masked_sketch, masked_sketch):
        # Fill in masked_sketch to make mask
        mask = self.fill_mask(masked_sketch)

        image_masked = image.clone()
        for i in range(image_masked.shape[-1]):
            image_masked[0,...,i] *= (1-mask.squeeze(0))

        new_mask = (ori_masked_sketch != mask).int().unsqueeze(-1)
        result = image * (1 - new_mask)

        return (mask, image_masked, result)

class MaskAlignedBbox4Inpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
                "linedistance" : ("INT", { "default": 0, "min": 0, "max": 100, "step": 0.5, })
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("MASK", "ORI_MASK", "MASKED_IMAGE", "MASKED_IMAGE_WITH_SKETCH")
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def connect_and_fill_contours(self, mask_np, linedistance=5):
        kernel = np.ones((linedistance, linedistance), np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled_mask_np = np.zeros_like(mask_np)
        cv2.drawContours(filled_mask_np, contours, -1, (1,), thickness=cv2.FILLED)

        return filled_mask_np

    def fill_mask(self, mask_tensor, linedistance=5):
        mask_np = mask_tensor.cpu().numpy()

        filled_mask_np = np.zeros_like(mask_np)

        for i in range(mask_np.shape[0]):
            mask_2d = mask_np[i]
            if mask_2d.ndim > 2:
                raise ValueError('Mask dimension is not 2D')
            connected_filled_mask_2d = self.connect_and_fill_contours(mask_2d.astype(np.uint8), linedistance)

            filled_mask_np[i] = connected_filled_mask_2d

        filled_mask_tensor = torch.from_numpy(filled_mask_np).to(mask_tensor.device, dtype=torch.float32)
        return filled_mask_tensor

    def execute(self, mask, padding, blur, linedistance,  image_optional=None):
        mask = (mask != 0.)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        # match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)

        filled_mask = self.fill_mask(mask, linedistance=linedistance)


        if padding > 0:
            kernel = torch.ones((1, 1, padding, padding), device=filled_mask.device, dtype=filled_mask.dtype)
            filled_mask = torch.nn.functional.conv2d(filled_mask.unsqueeze(1), kernel, padding=padding).squeeze(1)
            filled_mask = torch.clamp(filled_mask, 0, 1)

        image_optional_masked = image_optional.clone()
        for i in range(image_optional.shape[-1]):
            image_optional_masked[0,...,i] *= (1-filled_mask.squeeze(0))

        new_mask = (filled_mask != mask).int().unsqueeze(-1)
        result = image_optional * (1 - new_mask)

        return (mask, filled_mask.unsqueeze(0), image_optional_masked, result)

class MaskAlignedBbox4Inpainting2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
                "linedistance" : ("INT", { "default": 0, "min": 0, "max": 100, "step": 0.5, })
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("MASK", "ORI_MASK", "MASKED_IMAGE", "MASKED_IMAGE_WITH_SKETCH")
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def connect_and_fill_contours(self, mask_np, linedistance=5):
        kernel = np.ones((linedistance, linedistance), np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled_mask_np = np.zeros_like(mask_np)
        cv2.drawContours(filled_mask_np, contours, -1, (1,), thickness=cv2.FILLED)

        return filled_mask_np

    def fill_mask(self, mask_tensor, linedistance=5):
        mask_np = mask_tensor.cpu().numpy()

        filled_mask_np = np.zeros_like(mask_np)

        for i in range(mask_np.shape[0]):
            mask_2d = mask_np[i]
            if mask_2d.ndim > 2:
                raise ValueError('Mask dimension is not 2D')
            connected_filled_mask_2d = self.connect_and_fill_contours(mask_2d.astype(np.uint8), linedistance)

            filled_mask_np[i] = connected_filled_mask_2d

        filled_mask_tensor = torch.from_numpy(filled_mask_np).to(mask_tensor.device, dtype=torch.float32)
        return filled_mask_tensor

    def execute(self, mask, padding, blur, linedistance,  image_optional=None):
        mask = (mask != 0.)
        mask = mask.float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        # match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)

        filled_mask = mask
        if padding > 0:
            kernel = torch.ones((1, 1, padding, padding), device=filled_mask.device, dtype=filled_mask.dtype)
            filled_mask = torch.nn.functional.conv2d(filled_mask.unsqueeze(1), kernel, padding=padding).squeeze(1)
            filled_mask = torch.clamp(filled_mask, 0, 1)

        image_optional_masked = image_optional.clone()
        for i in range(image_optional.shape[-1]):
            image_optional_masked[0,...,i] *= (1-filled_mask.squeeze(0))

        new_mask = (filled_mask != mask).int().unsqueeze(-1)
        result = image_optional * (1 - new_mask)

        return (mask, filled_mask.unsqueeze(0), image_optional_masked, result)


class MaskSquareBbox4Inpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "ORI_MASK", "IMAGE", "x", "y", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, padding, blur, image_optional=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        # match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)

        _, y, x = torch.where(mask)
        x1 = max(0, x.min().item() - padding)
        x2 = min(mask.shape[2], x.max().item() + 1 + padding)
        y1 = max(0, y.min().item() - padding)
        y2 = min(mask.shape[1], y.max().item() + 1 + padding)

        # crop the mask
        mask = mask[:, y1:y2, x1:x2]
        ori_mask = torch.ones(image_optional.shape[1:3])
        ori_mask[y1:y2, x1:x2] = 0
        ori_mask = 1 - ori_mask

        image_optional_masked = image_optional.clone()
        for i in range(image_optional.shape[-1]):
            image_optional_masked[0,...,i] *= (1-ori_mask)
        return (mask, ori_mask.unsqueeze(0), image_optional_masked, x1, y1, x2 - x1, y2 - y1)


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
                              "scale": ("INT", {"default":3})
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    @torch.inference_mode()
    def upscale(self, upscale_model, image, scale):
        device = model_management.get_torch_device()
        model_management.soft_empty_cache()

        batch_size, height, width, channel = image.size()
        upscaled_images = torch.empty([batch_size, height*scale, width*scale, channel], device='cpu')

        for i in range(batch_size):
            in_img = image[i].unsqueeze(0).movedim(-1, -3).to(device)

            with torch.no_grad():
                output = upscale_model(in_img)
            s = torch.clamp(output.movedim(-3,-1), min=0, max=1.0)

            upscaled_images[i:i+1] = s.cpu()
            model_management.soft_empty_cache()

        model_management.soft_empty_cache()
        return (upscaled_images,)

class ConcaveHullImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "mask": ("MASK",),
                              "length_threshold": ("INT", {"default":10})
                              }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"

    @torch.inference_mode()
    def run(self, mask, length_threshold):
        processed_image = []
        for idx in range(mask.size(0)):
            twodmask = mask[idx]
            points = np.column_stack(np.where(twodmask == 1))
            concave_hull_mask = np.zeros_like(twodmask)

            if len(points) >= 3:
                idxes = concave_hull_indexes(points[:,:2], length_threshold=length_threshold)
                idxes = np.append(idxes, idxes[0])

                for f,t in zip(idxes[:-1], idxes[1:]):
                    seg = points[[f,t]]
                    cv2.line(concave_hull_mask,
                     (seg[:,1][0], seg[:,0][0]),
                     (seg[:,1][1], seg[:, 0][1]),
                     color=1, thickness=1)
            processed_image.append(concave_hull_mask)
        processed_image = torch.from_numpy(np.stack(processed_image))
        return (processed_image,)


import json
import os
import random

def read_json_file(file_path):
    """
    Reads a JSON file's content and returns it.
    Ensures content matches the expected format.
    """
    if not os.access(file_path, os.R_OK):
        print(f"Warning: No read permissions for file {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
            # Check if the content matches the expected format.
            if not all(['name' in item and 'prompt' in item and 'negative_prompt' in item for item in content]):
                print(f"Warning: Invalid content in file {file_path}")
                return None
            return content
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {str(e)}")
        return None


def read_sdxl_styles(json_data):
    """
    Returns style names from the provided JSON data.
    """
    if not isinstance(json_data, list):
        print("Error: input data must be a list")
        return []

    return [item['name'] for item in json_data if isinstance(item, dict) and 'name' in item]


def get_all_json_files(directory):
    """
    Returns all JSON files from the specified directory.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if
            file.endswith('.json') and os.path.isfile(os.path.join(directory, file))]


def load_styles_from_directory(directory):
    """
    Loads styles from all JSON files in the directory.
    Renames duplicate style names by appending a suffix.
    """
    json_files = get_all_json_files(directory)
    combined_data = []
    seen = set()

    for json_file in json_files:
        json_data = read_json_file(json_file)
        if json_data:
            for item in json_data:
                original_style = item['name']
                style = original_style
                suffix = 1
                while style in seen:
                    style = f"{original_style}_{suffix}"
                    suffix += 1
                item['name'] = style
                seen.add(style)
                combined_data.append(item)

    unique_style_names = [item['name'] for item in combined_data if isinstance(item, dict) and 'name' in item]

    return combined_data, unique_style_names


def validate_json_data(json_data):
    """
    Validates the structure of the JSON data.
    """
    if not isinstance(json_data, list):
        return False
    for template in json_data:
        if 'name' not in template or 'prompt' not in template:
            return False
    return True


def find_template_by_name(json_data, template_name):
    """
    Returns a template from the JSON data by name or None if not found.
    """
    for template in json_data:
        if template['name'] == template_name:
            return template
    return None


def split_template_advanced(template: str) -> tuple:
    """
    Splits a template into two parts based on a specific pattern.
    """
    if " . " in template:
        template_prompt_g, template_prompt_l = template.split(" . ", 1)
        template_prompt_g = template_prompt_g.strip()
        template_prompt_l = template_prompt_l.strip()
    else:
        template_prompt_g = template
        template_prompt_l = ""

    return template_prompt_g, template_prompt_l


def replace_prompts_in_template(template, positive_prompt, negative_prompt):
    """
    Replace the placeholders in a given template with the provided prompts.

    Args:
    - template (dict): The template containing prompt placeholders.
    - positive_prompt (str): The positive prompt to replace '{prompt}' in the template.
    - negative_prompt (str): The negative prompt to be combined with any existing negative prompt in the template.

    Returns:
    - tuple: A tuple containing the replaced positive and negative prompts.
    """
    positive_result = template['prompt'].replace('{prompt}', positive_prompt)

    json_negative_prompt = template.get('negative_prompt', "")
    negative_result = f"{json_negative_prompt}, {negative_prompt}" if json_negative_prompt and negative_prompt else json_negative_prompt or negative_prompt

    return positive_result, negative_result


def replace_prompts_in_template_advanced(template, positive_prompt_g, positive_prompt_l, negative_prompt,
                                         negative_prompt_to, copy_to_l):
    """
    Replace the placeholders in a given template with the provided prompts and split them accordingly.

    Args:
    - template (dict): The template containing prompt placeholders.
    - positive_prompt_g (str): The main positive prompt to replace '{prompt}' in the template.
    - positive_prompt_l (str): The auxiliary positive prompt to be combined in a specific manner.
    - negative_prompt (str): The negative prompt to be combined with any existing negative prompt in the template.
    - negative_prompt_to (str): The negative prompt destination {Both, G only, L only}.
    - copy_to_l (bool): Copy the G positive prompt to L.

    Returns:
    - tuple: A tuple containing the replaced main positive, auxiliary positive, combined positive,  main negative, auxiliary negative, and negative prompts.
    """
    template_prompt_g, template_prompt_l_template = split_template_advanced(template['prompt'])

    text_g_positive = template_prompt_g.replace("{prompt}", positive_prompt_g)

    text_l_positive = f"{template_prompt_l_template.replace('{prompt}', positive_prompt_g)}, {positive_prompt_l}" if template_prompt_l_template and positive_prompt_l else template_prompt_l_template.replace(
        '{prompt}', positive_prompt_g) or positive_prompt_l

    if copy_to_l and positive_prompt_g and "{prompt}" not in template_prompt_l_template:
        token_positive_g = list(map(lambda x: x.strip(), text_g_positive.split(",")))
        token_positive_l = list(map(lambda x: x.strip(), text_l_positive.split(",")))

        # deduplicate common prompt parts
        for token_g in token_positive_g:
            if token_g in token_positive_l:
                token_positive_l.remove(token_g)

        token_positive_g.extend(token_positive_l)

        text_l_positive = ", ".join(token_positive_g)

    text_positive = f"{text_g_positive} . {text_l_positive}" if text_l_positive else text_g_positive

    json_negative_prompt = template.get('negative_prompt', "")
    text_negative = f"{json_negative_prompt}, {negative_prompt}" if json_negative_prompt and negative_prompt else json_negative_prompt or negative_prompt

    text_g_negative = ""
    if negative_prompt_to in ("Both", "G only"):
        text_g_negative = text_negative

    text_l_negative = ""
    if negative_prompt_to in ("Both", "L only"):
        text_l_negative = text_negative

    return text_g_positive, text_l_positive, text_positive, text_g_negative, text_l_negative, text_negative


def read_sdxl_templates_replace_and_combine(json_data, template_name, positive_prompt, negative_prompt):
    """
    Find a specific template by its name, then replace and combine its placeholders with the provided prompts.

    Args:
    - json_data (list): The list of templates.
    - template_name (str): The name of the desired template.
    - positive_prompt (str): The positive prompt to replace placeholders.
    - negative_prompt (str): The negative prompt to be combined.

    Returns:
    - tuple: A tuple containing the replaced and combined positive and negative prompts.
    """
    if not validate_json_data(json_data):
        return positive_prompt, negative_prompt

    template = find_template_by_name(json_data, template_name)

    if template:
        return replace_prompts_in_template(template, positive_prompt, negative_prompt)
    else:
        return positive_prompt, negative_prompt


def read_sdxl_templates_replace_and_combine_advanced(json_data, template_name, positive_prompt_g, positive_prompt_l,
                                                     negative_prompt, negative_prompt_to, copy_to_l):
    """
    Find a specific template by its name, then replace and combine its placeholders with the provided prompts in an advanced manner.

    Args:
    - json_data (list): The list of templates.
    - template_name (str): The name of the desired template.
    - positive_prompt_g (str): The main positive prompt.
    - positive_prompt_l (str): The auxiliary positive prompt.
    - negative_prompt (str): The negative prompt to be combined.
    - negative_prompt_to (str): The negative prompt destination {Both, G only, L only}.
    - copy_to_l (bool): Copy the G positive prompt to L.

    Returns:
    - tuple: A tuple containing the replaced and combined main positive, auxiliary positive, combined positive, main negative, auxiliary negative, and negative prompts.
    """
    if not validate_json_data(json_data):
        return positive_prompt_g, positive_prompt_l, f"{positive_prompt_g} . {positive_prompt_l}", negative_prompt, negative_prompt, negative_prompt

    template = find_template_by_name(json_data, template_name)

    if template:
        return replace_prompts_in_template_advanced(template, positive_prompt_g, positive_prompt_l, negative_prompt,
                                                    negative_prompt_to, copy_to_l)
    else:
        return positive_prompt_g, positive_prompt_l, f"{positive_prompt_g} . {positive_prompt_l}", negative_prompt, negative_prompt, negative_prompt


class RandomPromptStyler:

    def __init__(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.json_data, styles = load_styles_from_directory(current_directory)
        pass

    @classmethod
    def INPUT_TYPES(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.json_data, self.styles = load_styles_from_directory(current_directory)

        return {
            "required": {
                "text_positive": ("STRING", {"default": "", "multiline": True}),
                "text_negative": ("STRING", {"default": "", "multiline": True}),
                "style": ((self.styles),),
                "random_styler": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "log_prompt": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "style_positive": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "style_negative": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ('STRING', 'STRING',)
    RETURN_NAMES = ('text_positive', 'text_negative',)
    FUNCTION = 'random_prompt_styler'
    CATEGORY = 'utils'

    def random_prompt_styler(self, text_positive, text_negative, style, random_styler, log_prompt, style_positive, style_negative, seed):
        if random_styler:
            style = random.choice(self.styles)
            text_positive_styled, text_negative_styled = read_sdxl_templates_replace_and_combine(self.json_data, style,
                                                                                                 text_positive,
                                                                                                 text_negative)
        else:
            text_positive_styled, text_negative_styled = read_sdxl_templates_replace_and_combine(self.json_data, style,
                                                                                             text_positive,
                                                                                             text_negative)

        # If style_negative is disabled, set text_negative_styled to text_negative
        if not style_positive:
            text_positive_styled = text_positive
            if log_prompt:
                print(f"style_positive: disabled")

        # If style_negative is disabled, set text_negative_styled to text_negative
        if not style_negative:
            text_negative_styled = text_negative
            if log_prompt:
                print(f"style_negative: disabled")

        # If logging is enabled (log_prompt is set to "Yes"),
        # print the style, positive and negative text, and positive and negative prompts to the console
        if log_prompt:
            print(f"style: {style}")
            print(f"text_positive: {text_positive}")
            print(f"text_negative: {text_negative}")
            print(f"text_positive_styled: {text_positive_styled}")
            print(f"text_negative_styled: {text_negative_styled}")

        return text_positive_styled, text_negative_styled



NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPromptStyler": "Random Prompt Styler",
}

NODE_CLASS_MAPPINGS = {
    "Mask Square bbox for Inpainting" : MaskSquareBbox4Inpainting,
    "Mask Aligned bbox for Inpainting": MaskAlignedBbox4Inpainting,
    "Mask Aligned bbox for Inpainting2": MaskAlignedBbox4Inpainting2,
    "Mask Aligned bbox for ConcaveHull" : MaskAlignedBbox4ConcaveHull,
    "One Image Compare": OneImageCompare,
    "Three Image Compare" : ThreeImageCompare,
    "Save Log Info": SaveLogInfo,
    "Add Human Styler" : HumanStyler,
    "Convert Monochrome" : ConvertMonochrome,
    "RT4KSR Loader" : RT4KSR_loader,
    "Upscale RT4SR" : Upscale_RT4SR,
    "RandomPromptStyler": RandomPromptStyler,
    "ConcaveHullImage": ConcaveHullImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save log info" : "Save Log Info",
    "RandomPromptStyler": "Random Prompt Styler"}

