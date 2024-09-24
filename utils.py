import numpy as np
import torch
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance

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

