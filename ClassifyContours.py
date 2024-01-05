import cv2
import numpy as np
from DefineModel import classify_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# label mappings to their unicode symbol
label_mapping = {
    '-' : '\002D'
}

# function to draw a symbol on a canvas
def draw_symbol(symbol, bounding_box, canvas):
    
    draw = ImageDraw.Draw(canvas)
    
    x, y, w, h = cv2.boundingRect(bounding_box)
    
    symbol_font_size = int(min(w, h))
    
    symbol_font = ImageFont.truetype("fonts/arial.ttf", symbol_font_size)
    
    text_position = (x, y)
    
    draw.text(text_position, symbol, fill='black', font=symbol_font)
    
    return canvas

# read input image
img = cv2.imread('input/target.jpg', cv2.IMREAD_GRAYSCALE)

# compute threshold
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
image_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

# generate contours
contours, ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# sort contours based on x-coordinate
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
print(f"number of contours: {len(contours)}")

# instantiate canvas to draw symbols on
canvas = Image.new('RGB', img.shape[:2], color='white')

# classify each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    symbol = img[y:y+h, x:x+w]
    symbol = Image.fromarray(symbol)
    label = classify_image('TinyVGG_0', symbol)
    canvas = draw_symbol(label, contour, canvas)
    
# display resulting canvas
canvas.save("output/transcribed.jpg")