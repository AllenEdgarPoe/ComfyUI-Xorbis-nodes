import cv2
import torchvision.transforms.v2 as T
from comfy import model_management
from .utils import *
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION
from .arch import *
from concave_hull import concave_hull_indexes
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing
import nodes
import base64
from io import BytesIO



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
                "concavehull_mask" : ("MASK",),
                "unfilled_mask": ("MASK",),
            },
        }
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    # MASK is for VAE Encode mask input
    # MASKED_IMAGE is for VAE Encode image input
    # MASKED_IMAGE_WITH_SKETCH is for ControlNet input
    RETURN_NAMES = ("MASK", "MASKED IMAGE", "UNFILLED MASKED IMAGE")
    FUNCTION = "execute"


    def execute(self, image, concavehull_mask, unfilled_mask):
        concavehull_mask = (concavehull_mask != 0.)
        concavehull_mask = concavehull_mask.float()

        if concavehull_mask.dim() == 2:
            concavehull_mask = concavehull_mask.unsqueeze(0)

        unfilled_mask = (unfilled_mask != 0.)
        unfilled_mask = unfilled_mask.float()

        if unfilled_mask.dim() == 2:
            unfilled_mask = unfilled_mask.unsqueeze(0)

        image_masked = image.clone()
        for i in range(image_masked.shape[-1]):
            image_masked[0,...,i] *= (1-concavehull_mask.squeeze(0))

        new_mask = (unfilled_mask != concavehull_mask).int().unsqueeze(-1)
        result = image * (1 - new_mask)

        return (concavehull_mask, image_masked, result)

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

        new_mask = (image_optional_masked != image_optional).int()
        # result = image_optional_masked * (1 - new_mask)
        result = torch.where(new_mask == 0, 0, image_optional)

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

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        for idx in range(mask.size(0)):
            twodmask = mask[idx]
            points = np.column_stack(np.where(twodmask != 0))
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


class InpaintCropXo:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_expand_pixels": ("INT", {"default": 20, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "context_expand_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
                "fill_mask_holes": ("BOOLEAN", {"default": True}),
                "blur_mask_pixels": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "blend_pixels": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "rescale_algorithm": (["nearest", "bilinear", "bicubic", "bislerp"], {"default": "bicubic"}),
                "mode": (["ranged size", "forced size", "free size"], {"default": "ranged size"}),
                "force_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),  # force
                "force_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),  # force
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01}),  # free
                "min_width": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),  # ranged
                "min_height": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),  # ranged
                "max_width": ("INT", {"default": 768, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),  # ranged
                "max_height": ("INT", {"default": 768, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),  # ranged
                "padding": ([8, 16, 32, 64, 128, 256, 512], {"default": 32}),  # free and ranged
                "drawing_mask" : ("MASK",)
            },
            "optional": {
                "optional_context_mask": ("MASK",),
            }
        }

    CATEGORY = "inpaint"

    RETURN_TYPES = ("STITCH", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask", "cropped_drawing_mask")

    FUNCTION = "inpaint_crop"

    def grow_and_blur_mask(self, mask, blur_pixels):
        if blur_pixels > 0.001:
            sigma = blur_pixels / 4
            growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in growmask:
                mask_np = m.numpy()
                kernel_size = math.ceil(sigma * 1.5 + 1)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                dilated_mask = grey_dilation(mask_np, footprint=kernel)
                output = dilated_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

            mask_np = mask.numpy()
            filtered_mask = gaussian_filter(mask_np, sigma=sigma)
            mask = torch.from_numpy(filtered_mask)
            mask = torch.clamp(mask, 0.0, 1.0)

        return mask

    def adjust_to_aspect_ratio(self, x_min, x_max, y_min, y_max, width, height, target_width, target_height):
        x_min_key, x_max_key, y_min_key, y_max_key = x_min, x_max, y_min, y_max

        # Calculate the current width and height
        current_width = x_max - x_min + 1
        current_height = y_max - y_min + 1

        # Calculate aspect ratios
        aspect_ratio = target_width / target_height
        current_aspect_ratio = current_width / current_height

        if current_aspect_ratio < aspect_ratio:
            # Adjust width to match target aspect ratio
            new_width = int(current_height * aspect_ratio)
            extend_x = (new_width - current_width)
            x_min = max(x_min - extend_x // 2, 0)
            x_max = min(x_max + extend_x // 2, width - 1)
        else:
            # Adjust height to match target aspect ratio
            new_height = int(current_width / aspect_ratio)
            extend_y = (new_height - current_height)
            y_min = max(y_min - extend_y // 2, 0)
            y_max = min(y_max + extend_y // 2, height - 1)

        return int(x_min), int(x_max), int(y_min), int(y_max)

    def adjust_to_preferred(self, x_min, x_max, y_min, y_max, width, height, preferred_x_start, preferred_x_end,
                            preferred_y_start, preferred_y_end):
        # Ensure the area is within preferred bounds as much as possible
        if preferred_x_start <= x_min and preferred_x_end >= x_max and preferred_y_start <= y_min and preferred_y_end >= y_max:
            return x_min, x_max, y_min, y_max

        # Shift x_min and x_max to fit within preferred bounds if possible
        if x_max - x_min + 1 <= preferred_x_end - preferred_x_start + 1:
            if x_min < preferred_x_start:
                x_shift = preferred_x_start - x_min
                x_min += x_shift
                x_max += x_shift
            elif x_max > preferred_x_end:
                x_shift = x_max - preferred_x_end
                x_min -= x_shift
                x_max -= x_shift

        # Shift y_min and y_max to fit within preferred bounds if possible
        if y_max - y_min + 1 <= preferred_y_end - preferred_y_start + 1:
            if y_min < preferred_y_start:
                y_shift = preferred_y_start - y_min
                y_min += y_shift
                y_max += y_shift
            elif y_max > preferred_y_end:
                y_shift = y_max - preferred_y_end
                y_min -= y_shift
                y_max -= y_shift

        return int(x_min), int(x_max), int(y_min), int(y_max)

    def apply_padding(self, min_val, max_val, max_boundary, padding):
        # Calculate the midpoint and the original range size
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2

        # Determine the smallest multiple of padding that is >= original_range_size
        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding

        # Calculate the new min and max values centered on the midpoint
        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1

        # Ensure the new max doesn't exceed the boundary
        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        # Ensure the range still ends on a multiple of padding
        # Adjust if the calculated range isn't feasible within the given constraints
        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        return new_min_val, new_max_val

    def inpaint_crop(self, image, mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels,
                     invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor,
                     padding, min_width, min_height, max_width, max_height, drawing_mask, optional_context_mask=None):
        if image.shape[0] > 1:
            assert mode == "forced size", "Mode must be 'forced size' when input is a batch of images"
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[
                0], "Batch size of optional_context_masks must be the same as images or None"


        result_stitch = {'x': [], 'y': [], 'original_image': [], 'cropped_mask_blend': [], 'rescale_x': [],
                         'rescale_y': [], 'start_x': [], 'start_y': [], 'initial_width': [], 'initial_height': []}
        results_image = []
        results_mask = []
        results_drawing_mask = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = None
            if optional_context_mask is not None:
                one_optional_context_mask = optional_context_mask[b].unsqueeze(0)

            stitch, cropped_image, cropped_mask, cropped_drawing_mask = self.inpaint_crop_single_image(one_image, one_mask,
                                                                                 context_expand_pixels,
                                                                                 context_expand_factor, fill_mask_holes,
                                                                                 blur_mask_pixels, invert_mask,
                                                                                 blend_pixels, mode, rescale_algorithm,
                                                                                 force_width, force_height,
                                                                                 rescale_factor, padding, min_width,
                                                                                 min_height, max_width, max_height, drawing_mask,
                                                                                 one_optional_context_mask)

            for key in result_stitch:
                result_stitch[key].append(stitch[key])
            cropped_image = cropped_image.squeeze(0)
            results_image.append(cropped_image)
            cropped_mask = cropped_mask.squeeze(0)
            results_mask.append(cropped_mask)
            cropped_drawing_mask = cropped_drawing_mask.squeeze(0)
            results_drawing_mask.append(cropped_drawing_mask)

        result_image = torch.stack(results_image, dim=0)
        result_mask = torch.stack(results_mask, dim=0)
        result_drawing_mask = torch.stack(results_drawing_mask, dim=0)

        return result_stitch, result_image, result_mask, result_drawing_mask

    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop_single_image(self, image, mask, context_expand_pixels, context_expand_factor, fill_mask_holes,
                                  blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width,
                                  force_height, rescale_factor, padding, min_width, min_height, max_width, max_height,
                                  drawing_mask, optional_context_mask=None):
        # Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"

        # Fill holes if requested
        if fill_mask_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Grow and blur mask if requested
        if blur_mask_pixels > 0.001:
            mask = self.grow_and_blur_mask(mask, blur_mask_pixels)

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Validate or initialize context mask
        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                assert False, "context_mask size must match image size"
        else:
            context_mask = optional_context_mask + mask
            context_mask = torch.clamp(context_mask, 0.0, 1.0)

        # Ensure mask dimensions match image dimensions except channels
        initial_batch, initial_height, initial_width, initial_channels = image.shape
        mask_batch, mask_height, mask_width = mask.shape
        context_mask_batch, context_mask_height, context_mask_width = context_mask.shape
        assert initial_height == mask_height and initial_width == mask_width, "Image and mask dimensions must match"
        assert initial_height == context_mask_height and initial_width == context_mask_width, "Image and context mask dimensions must match"

        # Extend image and masks to turn it into a big square in case the context area would go off bounds
        extend_y = (initial_width + 1) // 2  # Intended, extend height by width (turn into square)
        extend_x = (initial_height + 1) // 2  # Intended, extend width by height (turn into square)
        new_height = initial_height + 2 * extend_y
        new_width = initial_width + 2 * extend_x

        start_y = extend_y
        start_x = extend_x

        new_image = torch.zeros((initial_batch, new_height, new_width, initial_channels), dtype=image.dtype)
        new_image[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :] = image
        # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
        available_top = min(start_y, initial_height)
        available_bottom = min(new_height - (start_y + initial_height), initial_height)
        available_left = min(start_x, initial_width)
        available_right = min(new_width - (start_x + initial_width), initial_width)
        # Top
        new_image[:, start_y - available_top:start_y, start_x:start_x + initial_width, :] = torch.flip(
            image[:, :available_top, :, :], [1])
        # Bottom
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom,
        start_x:start_x + initial_width, :] = torch.flip(image[:, -available_bottom:, :, :], [1])
        # Left
        new_image[:, start_y:start_y + initial_height, start_x - available_left:start_x, :] = torch.flip(
            new_image[:, start_y:start_y + initial_height, start_x:start_x + available_left, :], [2])
        # Right
        new_image[:, start_y:start_y + initial_height,

        start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(
            new_image[:, start_y:start_y + initial_height,
            start_x + initial_width - available_right:start_x + initial_width, :], [2])
        # Top-left corner
        new_image[:, start_y - available_top:start_y, start_x - available_left:start_x, :] = torch.flip(
            new_image[:, start_y:start_y + available_top, start_x:start_x + available_left, :], [1, 2])
        # Top-right corner
        new_image[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right,
        :] = torch.flip(new_image[:, start_y:start_y + available_top,
                        start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
        # Bottom-left corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom,
        start_x - available_left:start_x, :] = torch.flip(
            new_image[:, start_y + initial_height - available_bottom:start_y + initial_height,
            start_x:start_x + available_left, :], [1, 2])
        # Bottom-right corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom,
        start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(
            new_image[:, start_y + initial_height - available_bottom:start_y + initial_height,
            start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])

        new_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype)
        new_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask

        new_drawing_mask = torch.ones((mask_batch, new_height, new_width), dtype=mask.dtype)
        new_drawing_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = drawing_mask

        blend_mask = torch.zeros((mask_batch, new_height, new_width),
                                 dtype=mask.dtype)  # assume zeros in extended image
        blend_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask

        new_context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=context_mask.dtype)
        new_context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = context_mask

        image = new_image
        mask = new_mask
        drawing_mask = new_drawing_mask
        context_mask = new_context_mask

        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        # If there are no non-zero indices in the context_mask, adjust context mask to the whole image
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            context_mask = torch.ones_like(image[:, :, :, 0])
            context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype)
            context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] += 1.0
            non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)

        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]

        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(max(y_size * (context_expand_factor - 1), context_expand_pixels, blend_pixels ** 1.5))
        x_grow = round(max(x_size * (context_expand_factor - 1), context_expand_pixels, blend_pixels ** 1.5))
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0

        # Adjust to preferred size
        if mode == 'forced size':
            # Sub case of ranged size.
            min_width = max_width = force_width
            min_height = max_height = force_height

        if mode == 'ranged size' or mode == 'forced size':
            assert max_width >= min_width, "max_width must be greater than or equal to min_width"
            assert max_height >= min_height, "max_height must be greater than or equal to min_height"
            # Ensure we set an aspect ratio supported by min_width, max_width, min_height, max_height
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1

            # Calculate aspect ratio of the selected area
            current_aspect_ratio = current_width / current_height

            # Calculate the aspect ratio bounds
            min_aspect_ratio = min_width / max_height
            max_aspect_ratio = max_width / min_height

            # Adjust target width and height based on aspect ratio bounds
            if current_aspect_ratio < min_aspect_ratio:
                # Adjust to meet minimum width constraint
                target_width = min(current_width, min_width)
                target_height = int(target_width / min_aspect_ratio)
                x_min, x_max, y_min, y_max = self.adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height,
                                                                         target_width, target_height)
                x_min, x_max, y_min, y_max = self.adjust_to_preferred(x_min, x_max, y_min, y_max, width, height,
                                                                      start_x, start_x + initial_width, start_y,
                                                                      start_y + initial_height)
            elif current_aspect_ratio > max_aspect_ratio:
                # Adjust to meet maximum width constraint
                target_height = min(current_height, max_height)
                target_width = int(target_height * max_aspect_ratio)
                x_min, x_max, y_min, y_max = self.adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height,
                                                                         target_width, target_height)
                x_min, x_max, y_min, y_max = self.adjust_to_preferred(x_min, x_max, y_min, y_max, width, height,
                                                                      start_x, start_x + initial_width, start_y,
                                                                      start_y + initial_height)
            else:
                # Aspect ratio is within bounds, keep the current size
                target_width = current_width
                target_height = current_height

            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1

            # Adjust to min and max sizes
            max_rescale_width = max_width / x_size
            max_rescale_height = max_height / y_size
            max_rescale_factor = min(max_rescale_width, max_rescale_height)
            rescale_factor = max_rescale_factor
            min_rescale_width = min_width / x_size
            min_rescale_height = min_height / y_size
            min_rescale_factor = min(min_rescale_width, min_rescale_height)
            rescale_factor = max(min_rescale_factor, rescale_factor)

        # Upscale image and masks if requested, they will be downsized at stitch phase
        if rescale_factor < 0.999 or rescale_factor > 1.001:
            samples = image
            samples = samples.movedim(-1, 1)
            width = math.floor(samples.shape[3] * rescale_factor)
            height = math.floor(samples.shape[2] * rescale_factor)
            samples = rescale(samples, width, height, rescale_algorithm)
            effective_upscale_factor_x = float(width) / float(original_width)
            effective_upscale_factor_y = float(height) / float(original_height)
            samples = samples.movedim(1, -1)
            image = samples

            samples = mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            mask = samples

            samples = drawing_mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            drawing_mask = samples

            samples = blend_mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            blend_mask = samples

            # Do math based on min,size instead of min,max to avoid rounding errors
            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            target_x_size = int(x_size * effective_upscale_factor_x)
            target_y_size = int(y_size * effective_upscale_factor_y)

            x_min = math.floor(x_min * effective_upscale_factor_x)
            x_max = x_min + target_x_size
            y_min = math.floor(y_min * effective_upscale_factor_y)
            y_max = y_min + target_y_size

        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Ensure width and height are within specified bounds, key for ranged and forced size
        if mode == 'ranged size' or mode == 'forced size':
            if x_size < min_width:
                x_max = min(x_max + (min_width - x_size), width - 1)
            elif x_size > max_width:
                x_max = x_min + max_width - 1

            if y_size < min_height:
                y_max = min(y_max + (min_height - y_size), height - 1)
            elif y_size > max_height:
                y_max = y_min + max_height - 1

        # Recalculate x_size and y_size after adjustments
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Pad area (if possible, i.e. if pad is smaller than width/height) to avoid the sampler returning smaller results
        if (mode == 'free size' or mode == 'ranged size') and padding > 1:
            x_min, x_max = self.apply_padding(x_min, x_max, width, padding)
            y_min, y_max = self.apply_padding(y_min, y_max, height, padding)

        # Ensure that context area doesn't go outside of the image
        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)

        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask[:, y_min:y_max + 1, x_min:x_max + 1]
        cropped_drawing_mask = drawing_mask[:, y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask_blend = blend_mask[:, y_min:y_max + 1, x_min:x_max + 1]

        # Grow and blur mask for blend if requested
        if blend_pixels > 0.001:
            cropped_mask_blend = self.grow_and_blur_mask(cropped_mask_blend, blend_pixels)

        # Return stitch (to be consumed by the class below), image, and mask
        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask_blend': cropped_mask_blend,
                  'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y, 'start_x': start_x,
                  'start_y': start_y, 'initial_width': initial_width, 'initial_height': initial_height}

        return (stitch, cropped_image, cropped_mask, cropped_drawing_mask)


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

class LoadData:
    def __init__(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.json_data, styles = load_styles_from_directory(current_directory)
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "img_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ('img', 'mask',)
    FUNCTION = 'load_data'

    def load_data(self, img_path):
        data = torch.load(img_path)
        img = data['image']
        mask = data['mask']

        return (img, mask)


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
    "ConcaveHullImage": ConcaveHullImage,
    "Inpaint Crop Xo" : InpaintCropXo,
    "LoadData" : LoadData
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save log info" : "Save Log Info",
    "RandomPromptStyler": "Random Prompt Styler"}

