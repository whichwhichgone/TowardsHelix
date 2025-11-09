import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random


TARGET_IMG_SIZE = 224
def text_to_image(instruction_path, output_path, subname):
    # Define colors
    background_colors = [
        (255, 200, 200),  # Light Red
        (200, 255, 200),  # Light Green
        (200, 200, 255),  # Light Blue
        (255, 255, 200),  # Light Yellow
        (255, 200, 255),  # Light Pink
        (200, 255, 255),  # Light Cyan
        (240, 240, 240),  # Light Gray
        (255, 230, 200),  # Light Orange
    ]
    text_colors = [
        (0, 0, 0),        # Black
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
        (0, 128, 0),      # Green
        (128, 0, 128),    # Purple
        (165, 42, 42),    # Brown
    ]
    
    # Define positions for text box (as percentages of image size)
    positions = []
    for x_pct in [0.1, 0.3, 0.5, 0.7]:
        for y_pct in [0.1, 0.3, 0.5, 0.7]:
            positions.append((x_pct, y_pct))
    
    # Define output directory
    output_dir = os.path.join(output_path, subname)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the YAML file
    with open(instruction_path, 'r') as file:
        tasks = json.load(file)
    
    # Initialize variables to track the longest instructions
    longest_char_instruction = ""
    longest_word_instruction = ""
    longest_char_count = 0
    longest_word_count = 0

    # Iterate through tasks to find the longest instructions
    for instruction in tasks:
        # Check for character count
        if len(instruction) > longest_char_count:
            longest_char_count = len(instruction)
            longest_char_instruction = instruction
        
        # Check for word count
        word_count = len(instruction.split())
        if word_count > longest_word_count:
            longest_word_count = word_count
            longest_word_instruction = instruction

    # Print statistics
    print(f"Longest instruction by character count ({longest_char_count} chars):")
    print(longest_char_instruction)
    print(f"Longest instruction by word count ({longest_word_count} words):")
    print(longest_word_instruction)

    # Try to load a font with a larger size
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        try:
            # For PIL versions that support setting size for default font
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            # If setting size isn't supported for default font
            font = ImageFont.load_default()
    
    # Function to estimate text width with fallbacks for different PIL versions
    def estimate_text_width(text, font):
        try:
            return font.getsize(text)[0]
        except AttributeError:
            try:
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0]
            except (AttributeError, TypeError):
                # Increase estimate for larger font
                return len(text) * 10  # Adjusted from 7 to 10 for larger font

    # Iterate through each task and its instructions
    for ids, instruction in enumerate(tasks):
        print(f"Processing task: {instruction}")
        
        style_index = 0
        for bg_color in background_colors:
            for text_color in text_colors:
                for position in positions:
                    # Create a base image
                    img_size = (TARGET_IMG_SIZE, TARGET_IMG_SIZE)
                    img = Image.new('RGB', img_size, color=bg_color)
                    
                    draw = ImageDraw.Draw(img)
                    
                    # Background box dimensions (70% width, 70% height)
                    bg_width = int(img_size[0] * 0.7)
                    bg_height = int(img_size[1] * 0.7)
                    
                    # Calculate box position
                    x_pos = int((img_size[0] - bg_width) * position[0])
                    y_pos = int((img_size[1] - bg_height) * position[1])
                    
                    # Ensure box stays within image boundaries
                    x_pos = max(0, min(x_pos, img_size[0] - bg_width))
                    y_pos = max(0, min(y_pos, img_size[1] - bg_height))
                    
                    # Draw the background box
                    draw.rectangle(
                        [(x_pos, y_pos), (x_pos + bg_width, y_pos + bg_height)], 
                        fill=bg_color
                    )
                    
                    # Wrap text to fit in the box
                    words = instruction.split()
                    lines = []
                    line = ""
                    
                    for word in words:
                        test_line = line + word + " "
                        text_width = estimate_text_width(test_line, font)
                        
                        if text_width < bg_width - 10:  # 10 pixels padding
                            line = test_line
                        else:
                            lines.append(line)
                            line = word + " "
                    
                    if line:
                        lines.append(line)
                    
                    # Draw text with increased line height (from 18 to 28)
                    text_x = x_pos + 10
                    text_y = y_pos + 10
                    line_height = int(font_size*1.3)  # Increased to accommodate larger font
                    
                    for i, line in enumerate(lines):
                        if text_y + (i+1) * line_height < y_pos + bg_height:
                            draw.text((text_x, text_y + i * line_height), line, fill=text_color, font=font)
                        else:
                            print(f"Warning: Line {i+1}/{len(lines)} truncated: '{line}'")
                    
                    # Save the image
                    img_path = os.path.join(output_dir, f"{instruction.replace(' ', '_')}_style{style_index}.png")
                    img.save(img_path)
                    
                    style_index += 1
                        
        print(f"Generated {style_index} instruction images in {output_dir}")

    return


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    text_to_image(
        instruction_path="/liujinxin/zhaowei/CogACT/vla/piper_res/ocr_instruction.json",
        output_path="/liujinxin/zhaowei/CogACT/vla/piper_res/utils",
        subname="piper_ocr",
    )
