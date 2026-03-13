from PIL import Image, ImageDraw, ImageOps
def create_single_image_zoom(img_path, output_path, crop_coords, zoom_scale=4, border_width=8):
    """
    Creates a side-by-side comparison of a full image and a zoomed-in detail.
    """
    # Load image
    img = Image.open(img_path).convert("RGB")
    
    # Define crop area (x, y, width, height)
    x, y, w, h = crop_coords
    crop_box = (x, y, x + w, y + h)

    # 1. Create and upscale the crop
    crop = img.crop(crop_box)
    zoom_size = (w * zoom_scale, h * zoom_scale)
    zoom = crop.resize(zoom_size, Image.LANCZOS)

    # 2. Add professional borders
    # Use a color that stands out (e.g., Yellow or Cyan)
    highlight_color = (255, 221, 0) # Gold/Yellow
    zoom = ImageOps.expand(zoom, border=border_width, fill=highlight_color)

    # 3. Draw the indicator box on the original image
    draw = ImageDraw.Draw(img)
    draw.rectangle(crop_box, outline=highlight_color, width=border_width)

    # 4. Combine side-by-side [Full Image | Zoomed Detail]
    padding = 20
    total_width = img.width + zoom.width + padding
    max_height = max(img.height, zoom.height)
    
    canvas = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    
    # Center vertically
    img_y = (max_height - img.height) // 2
    zoom_y = (max_height - zoom.height) // 2
    
    canvas.paste(img, (0, img_y))
    canvas.paste(zoom, (img.width + padding, zoom_y))

    canvas.save(output_path)
    print(f"Qualitative result saved to {output_path}")

# --- CONFIGURATION ---
# Example: Zooming into a specific detail on a DTU scan or ScanNet++ object
create_single_image_zoom(
    img_path="qualitative_results/base_vis_new.png", 
    output_path="qualitative_results/base_vis_new_zoom.png",
    crop_coords=(1000, 1, 300, 300), # Adjust these to hit your best detail!
    zoom_scale=3
)