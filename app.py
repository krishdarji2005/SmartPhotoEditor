from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for editor page
@app.route('/editor')
def editor():
    return render_template('editor.html')

# Image processing functions
def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Image Enhancement Functions

@app.route('/api/histogram_equalization', methods=['POST'])
def histogram_equalization():
    """Apply histogram equalization"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Convert back to BGR
        result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/negative_filter', methods=['POST'])
def negative_filter():
    """Apply negative transformation"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Digital negative
        result = 255 - img
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/threshold_filter', methods=['POST'])
def threshold_filter():
    """Apply thresholding"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        threshold_value = int(data.get('threshold', 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/negative', methods=['POST'])
def negative():
    """Apply negative transformation"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Digital negative
        result = 255 - img
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/log_transformation', methods=['POST'])
def log_transformation():
    """Apply log transformation"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Log transformation: s = c * log(1 + r)
        c = 255 / np.log(1 + np.max(img))
        result = c * np.log(1 + img)
        result = np.array(result, dtype=np.uint8)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/gamma_correction', methods=['POST'])
def gamma_correction():
    """Apply gamma correction"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        gamma = float(data.get('gamma', 1.0))
        
        # Gamma correction: s = c * r^gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        result = cv2.LUT(img, table)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/brightness_contrast', methods=['POST'])
def brightness_contrast():
    """Adjust brightness and contrast"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        brightness = int(data.get('brightness', 0))
        contrast = float(data.get('contrast', 1.0))
        
        # Apply brightness and contrast
        result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/threshold', methods=['POST'])
def threshold():
    """Apply thresholding"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        threshold_value = int(data.get('threshold', 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Filtering Functions

@app.route('/api/blur', methods=['POST'])
def blur():
    """Apply Gaussian blur"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        kernel_size = int(data.get('kernel_size', 5))
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur
        result = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/sharpen', methods=['POST'])
def sharpen():
    """Apply sharpening filter"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        result = cv2.filter2D(img, -1, kernel)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/edge_detection', methods=['POST'])
def edge_detection():
    """Apply edge detection (Laplacian)"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur first to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        # Convert back to BGR
        result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        print(f"Edge detection error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/noise_removal', methods=['POST'])
def noise_removal():
    """Remove salt and pepper noise using median filter"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        kernel_size = int(data.get('kernel_size', 5))
        
        # Apply median filter
        result = cv2.medianBlur(img, kernel_size)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Morphological Operations

@app.route('/api/erosion', methods=['POST'])
def erosion():
    """Apply erosion"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        kernel_size = int(data.get('kernel_size', 5))
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.erode(img, kernel, iterations=1)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/dilation', methods=['POST'])
def dilation():
    """Apply dilation"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        kernel_size = int(data.get('kernel_size', 5))
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.dilate(img, kernel, iterations=1)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/opening', methods=['POST'])
def opening():
    """Apply morphological opening"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        kernel_size = int(data.get('kernel_size', 5))
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/closing', methods=['POST'])
def closing():
    """Apply morphological closing"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        kernel_size = int(data.get('kernel_size', 5))
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Color Filters

@app.route('/api/vintage_filter', methods=['POST'])
def vintage_filter():
    """Apply vintage filter"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Vintage effect using color transformation
        rows, cols = img.shape[:2]
        
        # Create sepia effect
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        result = cv2.transform(img, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Add vignette
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
        
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        
        result[:,:,0] = result[:,:,0] * mask
        result[:,:,1] = result[:,:,1] * mask
        result[:,:,2] = result[:,:,2] * mask
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/cool_filter', methods=['POST'])
def cool_filter():
    """Apply cool (blue) filter"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Increase blue, decrease red
        result = img.copy()
        result[:,:,0] = np.clip(result[:,:,0] * 1.2, 0, 255)  # Blue
        result[:,:,2] = np.clip(result[:,:,2] * 0.8, 0, 255)  # Red
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/warm_filter', methods=['POST'])
def warm_filter():
    """Apply warm (red/orange) filter"""
    try:
        data = request.json
        img = base64_to_image(data['image'])
        
        # Increase red, decrease blue
        result = img.copy()
        result[:,:,2] = np.clip(result[:,:,2] * 1.2, 0, 255)  # Red
        result[:,:,0] = np.clip(result[:,:,0] * 0.8, 0, 255)  # Blue
        
        return jsonify({
            'success': True,
            'image': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)