import cv2
import numpy as np
import potrace

def trace_logo(input_path, output_path):
    # 1. Read the image
    image = cv2.imread(input_path)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Thresholding (Binarization) - Invert if text is black on white
    # We need the "ink" to be 1 and background to be 0 for Potrace
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Create a bitmap from the thresholded image
    bmp = potrace.Bitmap(thresh)
    
    # 5. Trace the bitmap to get paths
    path = bmp.trace()
    
    # 6. Write to SVG file
    with open(output_path, "w") as fp:
        fp.write(f'<svg width="{image.shape[1]}" height="{image.shape[0]}" viewBox="0 0 {image.shape[1]} {image.shape[0]}" xmlns="http://www.w3.org/2000/svg">')
        
        for curve in path:
            fp.write('<path d="')
            start = curve.start_point
            fp.write(f"M {start.x} {start.y} ")
            
            for segment in curve:
                if segment.is_corner:
                    c = segment.c
                    end = segment.end_point
                    fp.write(f"L {c.x} {c.y} L {end.x} {end.y} ")
                else:
                    c1 = segment.c1
                    c2 = segment.c2
                    end = segment.end_point
                    fp.write(f"C {c1.x} {c1.y} {c2.x} {c2.y} {end.x} {end.y} ")
            
            fp.write('Z" fill="black" stroke="none"/>')
            
        fp.write('</svg>')
    
    print(f"SVG saved to {output_path}")

# Usage
trace_logo("Crisitana Masi Logo.png", "CristianaMasi_Vector.svg")