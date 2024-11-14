import numpy as np
import torch
import os
import comfy.utils
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance

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

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def apply_resize_image(image: Image.Image, original_width, original_height, rounding_modulus, mode='scale',
                       supersample='true', factor: int = 2, width: int = 1024, height: int = 1024, resample='bicubic'):
    # Calculate the new width and height based on the given mode and parameters
    if mode == 'rescale':
        new_width, new_height = int(original_width * factor), int(original_height * factor)
    else:
        m = rounding_modulus
        original_ratio = original_height / original_width
        height = int(width * original_ratio)

        new_width = width if width % m == 0 else width + (m - width % m)
        new_height = height if height % m == 0 else height + (m - height % m)

    # Define a dictionary of resampling filters
    resample_filters = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'lanczos': 1}

    # Apply supersample
    if supersample == 'true':
        image = image.resize((new_width * 8, new_height * 8), resample=Image.Resampling(resample_filters[resample]))

    # Resize the image using the given resampling filter
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))

    return resized_image

def get_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)

    # Calculate the text width and height
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height

def justify_text(justify, img_width, line_width, margins):
    if justify == "left":
        text_plot_x = 0 + margins
    elif justify == "right":
        text_plot_x = img_width - line_width - margins
    elif justify == "center":
        text_plot_x = img_width / 2 - line_width / 2
    return text_plot_x

def align_text(align, img_height, text_height, text_pos_y, margins):
    if align == "center":
        text_plot_y = img_height / 2 - text_height / 2 + text_pos_y
    elif align == "top":
        text_plot_y = text_pos_y + margins
    elif align == "bottom":
        text_plot_y = img_height - text_height + text_pos_y - margins
    return text_plot_y


def draw_text(panel, text,
              font_name, font_size, font_color,
              font_outline_thickness, font_outline_color,
              bg_color,
              margins, line_spacing,
              position_x, position_y,
              align, justify,
              rotation_angle, rotation_options):
    # Create the drawing context
    draw = ImageDraw.Draw(panel)

    # Define font settings
    font_folder = "fonts"
    font_file = os.path.join(font_folder, font_name)
    resolved_font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), font_file)
    font = ImageFont.truetype(str(resolved_font_path), size=font_size)

    # Split the input text into lines
    text_lines = text.split('\n')

    # Calculate the size of the text plus padding for the tallest line
    max_text_width = 0
    max_text_height = 0

    for line in text_lines:
        # Calculate the width and height of the current line
        line_width, line_height = get_text_size(draw, line, font)

        line_height = line_height + line_spacing
        max_text_width = max(max_text_width, line_width)
        max_text_height = max(max_text_height, line_height)

    # Get the image center
    image_center_x = panel.width / 2
    image_center_y = panel.height / 2

    text_pos_y = position_y
    sum_text_plot_y = 0
    text_height = max_text_height * len(text_lines)

    for line in text_lines:
        # Calculate the width and height of the current line
        line_width, line_height = get_text_size(draw, line, font)

        # Get the text x and y positions for each line
        text_plot_x = position_x + justify_text(justify, panel.width, line_width, margins)
        text_plot_y = align_text(align, panel.height, text_height, text_pos_y, margins)

        # Add the current line to the text mask
        draw.text((text_plot_x, text_plot_y), line, fill=font_color, font=font, stroke_width=font_outline_thickness,
                  stroke_fill=font_outline_color)

        text_pos_y += max_text_height  # Move down for the next line
        sum_text_plot_y += text_plot_y  # Sum the y positions

    text_center_x = text_plot_x + max_text_width / 2
    text_center_y = sum_text_plot_y / len(text_lines)

    if rotation_options == "text center":
        rotated_panel = panel.rotate(rotation_angle, center=(text_center_x, text_center_y), resample=Image.BILINEAR)
    elif rotation_options == "image center":
        rotated_panel = panel.rotate(rotation_angle, center=(image_center_x, image_center_y), resample=Image.BILINEAR)

    return rotated_panel

def text_panel(image_width, image_height, text,
               font_name, font_size, font_color,
               font_outline_thickness, font_outline_color,
               background_color,
               margins, line_spacing,
               position_x, position_y,
               align, justify,
               rotation_angle, rotation_options):
    """
    Create an image with text overlaid on a background.

    Returns:
    PIL.Image.Image: Image with text overlaid on the background.
    """

    # Create PIL images for the text and background layers and text mask
    size = (image_width, image_height)
    panel = Image.new('RGB', size, background_color)

    # Draw the text on the text mask
    image_out = draw_text(panel, text,
                          font_name, font_size, font_color,
                          font_outline_thickness, font_outline_color,
                          background_color,
                          margins, line_spacing,
                          position_x, position_y,
                          align, justify,
                          rotation_angle, rotation_options)

    return image_out

def rescale(samples, width, height, algorithm):
    if algorithm == "nearest":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="nearest")
    elif algorithm == "bilinear":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear")
    elif algorithm == "bicubic":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bicubic")
    elif algorithm == "bislerp":
        return comfy.utils.bislerp(samples, width, height)
    return samples


def combine_images(images, layout_direction='horizontal'):
    """
    Combine a list of PIL Image objects either horizontally or vertically.

    Args:
    images (list of PIL.Image.Image): List of PIL Image objects to combine.
    layout_direction (str): 'horizontal' for horizontal layout, 'vertical' for vertical layout.

    Returns:
    PIL.Image.Image: Combined image.
    """

    if layout_direction == 'horizontal':
        combined_width = sum(image.width for image in images)
        combined_height = max(image.height for image in images)
    else:
        combined_width = max(image.width for image in images)
        combined_height = sum(image.height for image in images)

    combined_image = Image.new('RGB', (combined_width, combined_height))

    x_offset = 0
    y_offset = 0  # Initialize y_offset for vertical layout
    for image in images:
        combined_image.paste(image, (x_offset, y_offset))
        if layout_direction == 'horizontal':
            x_offset += image.width
        else:
            y_offset += image.height

    return combined_image

