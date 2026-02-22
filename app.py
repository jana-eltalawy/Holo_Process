from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Color blindness simulation matrices (linear RGB)
COLOR_BLINDNESS_MATRICES = {
    'protanopia': np.array([
        [0.567, 0.433, 0.0],
        [0.558, 0.442, 0.0],
        [0.0,   0.242, 0.758]
    ], dtype=np.float32),
    
    'deuteranopia': np.array([
        [0.625, 0.375, 0.0],
        [0.700, 0.300, 0.0],
        [0.0,   0.300, 0.700]
    ], dtype=np.float32),
    
    'tritanopia': np.array([
        [0.950, 0.050, 0.0],
        [0.0,   0.433, 0.567],
        [0.0,   0.475, 0.525]
    ], dtype=np.float32)
}

class ImageProcessor:
    """Advanced image processing class with color blindness simulation"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def apply_color_matrix(self, image, matrix):
        """Apply 3x3 color transformation matrix to image"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Apply matrix transformation
            transformed = cv2.transform(image, matrix)
            # Ensure values are in valid range
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)
            return transformed
        return image
    
    def simulate_color_blindness(self, image, type_name):
        """Simulate color vision deficiency"""
        if type_name not in COLOR_BLINDNESS_MATRICES:
            return image
        
        matrix = COLOR_BLINDNESS_MATRICES[type_name]
        return self.apply_color_matrix(image, matrix)
    
    def apply_daltonization(self, image, type_name):
        """Apply Daltonization correction for color blindness"""
        original = image.copy()
        
        # Simulate the color blindness first
        simulated = self.simulate_color_blindness(original, type_name)
        
        # Calculate error
        error = original.astype(np.float32) - simulated.astype(np.float32)
        
        # Apply correction based on type
        if type_name in ['protanopia', 'deuteranopia']:
            # For red-green deficiency, shift lost info into blue channel
            corrected = original.astype(np.float32).copy()
            corrected[:, :, 2] += 0.7 * error[:, :, 0] + 0.3 * error[:, :, 1]
        elif type_name == 'tritanopia':
            # For blue deficiency, shift lost info into red and green channels
            corrected = original.astype(np.float32).copy()
            corrected[:, :, 0] += 0.5 * error[:, :, 2]
            corrected[:, :, 1] += 0.5 * error[:, :, 2]
        else:
            corrected = original
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def apply_grayscale(self, image):
        """Convert image to grayscale using luminance formula"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image
    
    def apply_sepia(self, image):
        """Apply sepia tone effect"""
        if len(image.shape) == 3:
            # Sepia transformation matrix
            sepia_filter = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ], dtype=np.float32)
            
            sepia = cv2.transform(image, sepia_filter)
            return np.clip(sepia, 0, 255).astype(np.uint8)
        return image
    
    def apply_invert(self, image):
        """Invert colors"""
        return cv2.bitwise_not(image)
    
    def adjust_brightness(self, image, value=40):
        """Adjust brightness"""
        return cv2.convertScaleAbs(image, alpha=1.0, beta=value)
    
    def adjust_contrast(self, image, alpha=1.4):
        """Adjust contrast"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=128*(1-alpha))
    
    def adjust_saturation(self, image, factor=1.5):
        """Adjust saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], factor)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def apply_gamma_correction(self, image, gamma=1.2):
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def rotate_image(self, image, angle=45):
        """Rotate image"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (width, height))
        return rotated
    
    def translate_image(self, image, tx=20, ty=20):
        """Translate image"""
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def shear_image(self, image, shear_factor=0.15):
        """Apply shearing transformation"""
        height, width = image.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(image, M, (width, height))
    
    def apply_blur(self, image, kernel_size=3):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_sharpen(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def apply_edge_detection(self, image):
        """Apply edge detection using Laplacian"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def apply_emboss(self, image):
        """Apply emboss effect"""
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(image, -1, kernel)
    
    def apply_median_filter(self, image, kernel_size=3):
        """Apply median filter"""
        return cv2.medianBlur(image, kernel_size)
    
    def add_noise(self, image, noise_factor=25):
        """Add salt and pepper noise"""
        noise = np.random.normal(0, noise_factor, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def apply_morphology(self, image, operation='erosion'):
        """Apply morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        kernel = np.ones((3,3), np.uint8)
        
        if operation == 'erosion':
            result = cv2.erode(gray, kernel, iterations=1)
        elif operation == 'dilation':
            result = cv2.dilate(gray, kernel, iterations=1)
        else:
            result = gray
        
        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
    
    def apply_histogram_equalization(self, image):
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def apply_cartoon_effect(self, image):
        """Apply cartoon-like effect"""
        # Apply median filter to reduce noise
        median = cv2.medianBlur(image, 19)
        
        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine median filter with edges
        cartoon = cv2.bitwise_and(median, edges)
        return cartoon
    
    def calculate_histogram(self, image):
        """Calculate image histogram"""
        if len(image.shape) == 3:
            # Calculate for each channel
            histograms = {
                'gray': cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256]),
                'red': cv2.calcHist([image], [2], None, [256], [0,256]),
                'green': cv2.calcHist([image], [1], None, [256], [0,256]),
                'blue': cv2.calcHist([image], [0], None, [256], [0,256])
            }
        else:
            histograms = {
                'gray': cv2.calcHist([image], [0], None, [256], [0,256])
            }
        return histograms

# Initialize processor
processor = ImageProcessor()

@app.route('/')
def index():
    """Serve the main HTML file"""
    try:
        with open("template\index.html", 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Holo_Process</title></head>
        <body>
            <h1>Holo_Process Server Running</h1>
            <p>Server is running. Please ensure index.html is in the correct location.</p>
            <p>Available endpoints:</p>
            <ul>
                <li>/process - Apply image filters</li>
                <li>/histogram - Calculate image histograms</li>
                <li>/download - Download processed image</li>
            </ul>
        </body>
        </html>
        """

@app.route('/process', methods=['POST'])
def process_image():
    """Process image with selected filter"""
    try:
        # Get image data from request
        data = request.json
        image_data = data.get('image_data')
        filter_type = data.get('filter')
        
        if not image_data or not filter_type:
            return jsonify({'error': 'Missing image data or filter type'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Apply filter
        processed_image = apply_filter(filter_type, image)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.png', processed_image)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f'data:image/png;base64,{processed_base64}',
            'filter_applied': filter_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/histogram', methods=['POST'])
def calculate_histogram():
    """Calculate histogram for image"""
    try:
        data = request.json
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Decode and process image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Calculate histogram
        histograms = processor.calculate_histogram(image)
        
        # Convert numpy arrays to lists for JSON serialization
        hist_data = {}
        for key, hist in histograms.items():
            hist_data[key] = hist.flatten().tolist()
        
        return jsonify(histograms=hist_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download_image():
    """Download processed image"""
    try:
        
        if not image_data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Decode and save image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            tmp_file.write(image_bytes)
            tmp_file.flush()
            
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f'processed_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                mimetype='image/png'
            )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def apply_filter(filter_type, image):
    """Apply specified filter to image"""
    if filter_type == 'grayscale':
        return processor.apply_grayscale(image)
    elif filter_type == 'sepia':
        return processor.apply_sepia(image)
    elif filter_type == 'invert':
        return processor.apply_invert(image)
    elif filter_type == 'brightness':
        return processor.adjust_brightness(image, 40)
    elif filter_type == 'contrast':
        return processor.adjust_contrast(image, 1.4)
    elif filter_type == 'saturation':
        return processor.adjust_saturation(image, 1.5)
    elif filter_type == 'gamma':
        return processor.apply_gamma_correction(image, 1.2)
    elif filter_type == 'gamma_low':
        return processor.apply_gamma_correction(image, 0.5)
    elif filter_type == 'gamma_high':
        return processor.apply_gamma_correction(image, 2.0)
    elif filter_type == 'rotation':
        return processor.rotate_image(image, 45)
    elif filter_type == 'translation':
        return processor.translate_image(image, 20, 20)
    elif filter_type == 'shearing':
        return processor.shear_image(image, 0.15)
    elif filter_type == 'blur':
        return processor.apply_blur(image, 5)
    elif filter_type == 'sharpen':
        return processor.apply_sharpen(image)
    elif filter_type == 'edge':
        return processor.apply_edge_detection(image)
    elif filter_type == 'emboss':
        return processor.apply_emboss(image)
    elif filter_type == 'median':
        return processor.apply_median_filter(image, 3)
    elif filter_type == 'noise':
        return processor.add_noise(image, 25)
    elif filter_type == 'erosion':
        return processor.apply_morphology(image, 'erosion')
    elif filter_type == 'dilation':
        return processor.apply_morphology(image, 'dilation')
    elif filter_type == 'histogram_eq':
        return processor.apply_histogram_equalization(image)
    elif filter_type == 'cartoon':
        return processor.apply_cartoon_effect(image)
    elif filter_type == 'cvd_protanopia':
        return processor.simulate_color_blindness(image, 'protanopia')
    elif filter_type == 'cvd_deuteranopia':
        return processor.simulate_color_blindness(image, 'deuteranopia')
    elif filter_type == 'cvd_tritanopia':
        return processor.simulate_color_blindness(image, 'tritanopia')
    elif filter_type == 'cvd_protanopia_corr':
        return processor.apply_daltonization(image, 'protanopia')
    elif filter_type == 'cvd_deuteranopia_corr':
        return processor.apply_daltonization(image, 'deuteranopia')
    elif filter_type == 'cvd_tritanopia_corr':
        return processor.apply_daltonization(image, 'tritanopia')
    else:
        return image

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Holo_Process Image Processing'})

@app.route('/info')
def app_info():
    """Application information"""
    return jsonify({
        'name': 'Holo_Process',
        'version': '2.0.0',
        'features': [
            'Basic Filters (Grayscale, Sepia, Invert)',
            'Brightness/Contrast/Saturation',
            'Gamma Correction',
            'Geometric Transformations',
            'Convolution Filters',
            'Morphological Operations',
            'Color Blindness Simulator & Corrector',
            'Histogram Analysis'
        ],
        'color_blindness_support': [
            'Protanopia (red-blind)',
            'Deuteranopia (green-blind)', 
            'Tritanopia (blue-blind)',
            'Daltonization Correction'
        ]
    })

if __name__ == '__main__':
    print("🎨 Starting Holo_Process Image Processing Server...")
    print("📊 Color Blindness Simulator & Corrector enabled")
    print("🚀 Server will be available at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)