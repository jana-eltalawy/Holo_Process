
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

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# =============================================================================
# COLOR BLINDNESS SIMULATION MATRICES (SCIENTIFICALLY ACCURATE)
# =============================================================================

# These matrices are based on academic research and are widely used in 
# professional color blindness simulators like Coblis and Vischeck
COLOR_BLINDNESS_MATRICES = {
    'protanopia': np.array([
        [0.567, 0.433, 0.0],   # Red channel transformation
        [0.558, 0.442, 0.0],   # Green channel transformation  
        [0.0,   0.242, 0.758]  # Blue channel transformation
    ], dtype=np.float32),
    
    'deuteranopia': np.array([
        [0.625, 0.375, 0.0],   # Most common type (~6% of males)
        [0.700, 0.300, 0.0],   # Red-green confusion
        [0.0,   0.300, 0.700]  # Preserves some blue information
    ], dtype=np.float32),
    
    'tritanopia': np.array([
        [0.950, 0.050, 0.0],   # Rare type (~0.001% of population)
        [0.0,   0.433, 0.567], # Blue-yellow confusion
        [0.0,   0.475, 0.525]  # Blue cone deficiency
    ], dtype=np.float32)
}

# =============================================================================
# ADVANCED IMAGE PROCESSING CLASS
# =============================================================================

class ImageProcessor:
    """
    Comprehensive image processing class with color blindness accessibility
    
    This class provides both basic image processing operations and advanced
    color blindness simulation and correction capabilities.
    """
    
    def __init__(self):
        """Initialize the image processor with supported formats"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def apply_color_matrix(self, image, matrix):
        """
        Apply 3x3 color transformation matrix to image
        
        Args:
            image: Input image as numpy array
            matrix: 3x3 transformation matrix
            
        Returns:
            Transformed image with applied matrix
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Apply matrix transformation using OpenCV
            transformed = cv2.transform(image, matrix)
            # Ensure values are in valid range [0, 255]
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)
            return transformed
        return image
    
    def simulate_color_blindness(self, image, type_name):
        """
        Simulate color vision deficiency using scientific matrices
        
        Args:
            image: Input image as numpy array
            type_name: Type of color blindness ('protanopia', 'deuteranopia', 'tritanopia')
            
        Returns:
            Image as it would appear to someone with the specified color vision deficiency
        """
        if type_name not in COLOR_BLINDNESS_MATRICES:
            return image
        
        matrix = COLOR_BLINDNESS_MATRICES[type_name]
        return self.apply_color_matrix(image, matrix)
    
    def apply_daltonization(self, image, type_name):
        """
        Apply Daltonization correction for color blindness
        
        This algorithm enhances lost color information by shifting it to
        visible channels, making the image more accessible.
        
        Args:
            image: Input image as numpy array
            type_name: Type of color blindness to correct
            
        Returns:
            Corrected image with enhanced color information
        """
        original = image.copy()
        
        # Step 1: Simulate the color blindness first
        simulated = self.simulate_color_blindness(original, type_name)
        
        # Step 2: Calculate error between original and simulated
        error = original.astype(np.float32) - simulated.astype(np.float32)
        
        # Step 3: Apply correction based on deficiency type
        if type_name in ['protanopia', 'deuteranopia']:
            # For red-green deficiency, shift lost info into blue channel
            # This is based on the principle that blue channel is less affected
            corrected = original.astype(np.float32).copy()
            corrected[:, :, 2] += 0.7 * error[:, :, 0] + 0.3 * error[:, :, 1]
        elif type_name == 'tritanopia':
            # For blue deficiency, shift lost info into red and green channels
            corrected = original.astype(np.float32).copy()
            corrected[:, :, 0] += 0.5 * error[:, :, 2]
            corrected[:, :, 1] += 0.5 * error[:, :, 2]
        else:
            corrected = original
        
        # Ensure values are in valid range
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    # =============================================================================
    # BASIC IMAGE PROCESSING OPERATIONS
    # =============================================================================
    
    def apply_grayscale(self, image):
        """Convert image to grayscale using luminance formula"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image
    
    def apply_sepia(self, image):
        """Apply sepia tone effect using transformation matrix"""
        if len(image.shape) == 3:
            # Sepia transformation matrix (widely used standard)
            sepia_filter = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ], dtype=np.float32)
            
            sepia = cv2.transform(image, sepia_filter)
            return np.clip(sepia, 0, 255).astype(np.uint8)
        return image
    
    def apply_invert(self, image):
        """Invert colors (negative effect)"""
        return cv2.bitwise_not(image)
    
    def adjust_brightness(self, image, value=40):
        """Adjust image brightness"""
        return cv2.convertScaleAbs(image, alpha=1.0, beta=value)
    
    def adjust_contrast(self, image, alpha=1.4):
        """Adjust image contrast"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=128*(1-alpha))
    
    def adjust_saturation(self, image, factor=1.5):
        """Adjust color saturation using HSV color space"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], factor)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def apply_gamma_correction(self, image, gamma=1.2):
        """Apply gamma correction for tone mapping"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    # =============================================================================
    # GEOMETRIC TRANSFORMATIONS
    # =============================================================================
    
    def rotate_image(self, image, angle=45):
        """Rotate image by specified angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation with border handling
        rotated = cv2.warpAffine(image, M, (width, height))
        return rotated
    
    def translate_image(self, image, tx=20, ty=20):
        """Translate image by specified offsets"""
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def shear_image(self, image, shear_factor=0.15):
        """Apply shearing transformation"""
        height, width = image.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(image, M, (width, height))
    
    # =============================================================================
    # CONVOLUTION AND FILTERING OPERATIONS
    # =============================================================================
    
    def apply_blur(self, image, kernel_size=3):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_sharpen(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def apply_edge_detection(self, image):
        """Apply edge detection using Laplacian operator"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def apply_emboss(self, image):
        """Apply emboss effect using custom kernel"""
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(image, -1, kernel)
    
    def apply_median_filter(self, image, kernel_size=3):
        """Apply median filter for noise reduction"""
        return cv2.medianBlur(image, kernel_size)
    
    def add_noise(self, image, noise_factor=25):
        """Add salt and pepper noise to image"""
        noise = np.random.normal(0, noise_factor, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    # =============================================================================
    # MORPHOLOGICAL OPERATIONS
    # =============================================================================
    
    def apply_morphology(self, image, operation='erosion'):
        """Apply morphological operations (erosion/dilation)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 3x3 structuring element
        kernel = np.ones((3,3), np.uint8)
        
        if operation == 'erosion':
            result = cv2.erode(gray, kernel, iterations=1)
        elif operation == 'dilation':
            result = cv2.dilate(gray, kernel, iterations=1)
        else:
            result = gray
        
        # Convert back to BGR if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
    
    def apply_histogram_equalization(self, image):
        """Apply histogram equalization for contrast enhancement"""
        if len(image.shape) == 3:
            # Use LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def apply_cartoon_effect(self, image):
        """Apply cartoon-like effect using median filter and edge detection"""
        # Step 1: Apply median filter to reduce noise
        median = cv2.medianBlur(image, 19)
        
        # Step 2: Edge detection using adaptive threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Step 3: Combine median filter with edges
        cartoon = cv2.bitwise_and(median, edges)
        return cartoon
    
    # =============================================================================
    # HISTOGRAM ANALYSIS
    # =============================================================================
    
    def calculate_histogram(self, image):
        """
        Calculate comprehensive histogram analysis
        
        Returns:
            Dictionary with grayscale and RGB channel histograms
        """
        if len(image.shape) == 3:
            # Calculate for each channel
            histograms = {
                'gray': cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256]),
                'red': cv2.calcHist([image], [2], None, [256], [0,256]),     # OpenCV uses BGR
                'green': cv2.calcHist([image], [1], None, [256], [0,256]),
                'blue': cv2.calcHist([image], [0], None, [256], [0,256])
            }
        else:
            histograms = {
                'gray': cv2.calcHist([image], [0], None, [256], [0,256])
            }
        return histograms

# =============================================================================
# INITIALIZE IMAGE PROCESSOR
# =============================================================================

processor = ImageProcessor()

# =============================================================================
# FLASK ROUTES AND API ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """Serve the main HTML interface"""
    try:
        # Try to serve the enhanced HTML file
        with open('/workspace/index_enhanced.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to original HTML file
        try:
            with open('/workspace/user_input_files/index.html', 'r') as f:
                return f.read()
        except FileNotFoundError:
            # Return basic server information
            return """
            <!DOCTYPE html>
            <html>
            <head><title>Holo_Process Server</title></head>
            <body style="font-family: Arial, sans-serif; background: #1f2937; color: white; padding: 20px;">
                <h1>🎨 Holo_Process Server Running</h1>
                <p>Advanced Image Processing with Color Blindness Accessibility</p>
                <hr>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><strong>/process</strong> - Apply image filters</li>
                    <li><strong>/histogram</strong> - Calculate image histograms</li>
                    <li><strong>/download</strong> - Download processed images</li>
                    <li><strong>/health</strong> - Server health check</li>
                    <li><strong>/info</strong> - Application information</li>
                </ul>
                <hr>
                <h2>Features:</h2>
                <ul>
                    <li>✅ Color Blindness Simulator (Protanopia, Deuteranopia, Tritanopia)</li>
                    <li>✅ Daltonization Correction</li>
                    <li>✅ Advanced Image Filters</li>
                    <li>✅ Real-time Histogram Analysis</li>
                    <li>✅ Client/Server Dual Mode</li>
                </ul>
                <p><em>Ensure HTML files are in the correct location for full functionality.</em></p>
            </body>
            </html>
            """

@app.route('/process', methods=['POST'])
def process_image():
    """
    Process image with selected filter
    
    Expected JSON:
    {
        "image_data": "data:image/jpeg;base64,...",
        "filter": "filter_name"
    }
    
    Returns:
    {
        "processed_image": "data:image/png;base64,...",
        "filter_applied": "filter_name"
    }
    """
    try:
        # Get request data
        data = request.json
        image_data = data.get('image_data')
        filter_type = data.get('filter')
        
        if not image_data or not filter_type:
            return jsonify({'error': 'Missing image data or filter type'}), 400
        
        # Decode base64 image data
        # Remove data:image/jpeg;base64, prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data or format not supported'}), 400
        
        # Apply the specified filter
        processed_image = apply_filter(filter_type, image)
        
        # Convert back to base64 for transmission
        _, buffer = cv2.imencode('.png', processed_image)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f'data:image/png;base64,{processed_base64}',
            'filter_applied': filter_type,
            'processing_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/histogram', methods=['POST'])
def calculate_histogram():
    """
    Calculate histogram for uploaded image
    
    Expected JSON:
    {
        "image_data": "data:image/jpeg;base64,..."
    }
    
    Returns:
    {
        "histograms": {
            "gray": [256 values],
            "red": [256 values],
            "green": [256 values], 
            "blue": [256 values]
        }
    }
    """
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
        
        # Calculate comprehensive histogram
        histograms = processor.calculate_histogram(image)
        
        # Convert numpy arrays to lists for JSON serialization
        hist_data = {}
        for key, hist in histograms.items():
            hist_data[key] = hist.flatten().tolist()
        
        return jsonify(histograms=hist_data)
        
    except Exception as e:
        return jsonify({'error': f'Histogram calculation failed: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download_image():
    """
    Download processed image as file
    
    Expected JSON:
    {
        "image_data": "data:image/png;base64,..."
    }
    
    Returns:
    File download with appropriate headers
    """
    try:
        data = request.json
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Decode and save image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            tmp_file.write(image_bytes)
            tmp_file.flush()
            
            # Generate filename with timestamp
            filename = f'processed_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=filename,
                mimetype='image/png'
            )
            
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

def apply_filter(filter_type, image):
    """
    Apply specified filter to image
    
    Args:
        filter_type: Name of filter to apply
        image: Input image as numpy array
        
    Returns:
        Processed image as numpy array
    """
    # Map filter names to processing functions
    filter_mapping = {
        # Basic filters
        'grayscale': processor.apply_grayscale,
        'sepia': processor.apply_sepia,
        'invert': processor.apply_invert,
        'brightness': lambda img: processor.adjust_brightness(img, 40),
        'contrast': lambda img: processor.adjust_contrast(img, 1.4),
        'saturation': lambda img: processor.adjust_saturation(img, 1.5),
        
        # Gamma correction
        'gamma': lambda img: processor.apply_gamma_correction(img, 1.2),
        'gamma_low': lambda img: processor.apply_gamma_correction(img, 0.5),
        'gamma_high': lambda img: processor.apply_gamma_correction(img, 2.0),
        
        # Geometric transformations
        'rotation': lambda img: processor.rotate_image(img, 45),
        'translation': lambda img: processor.translate_image(img, 20, 20),
        'shearing': lambda img: processor.shear_image(img, 0.15),
        
        # Convolution filters
        'blur': lambda img: processor.apply_blur(img, 5),
        'sharpen': processor.apply_sharpen,
        'edge': processor.apply_edge_detection,
        'emboss': processor.apply_emboss,
        'median': lambda img: processor.apply_median_filter(img, 3),
        'noise': lambda img: processor.add_noise(img, 25),
        
        # Morphological operations
        'erosion': lambda img: processor.apply_morphology(img, 'erosion'),
        'dilation': lambda img: processor.apply_morphology(img, 'dilation'),
        'histogram_eq': processor.apply_histogram_equalization,
        'cartoon': processor.apply_cartoon_effect,
        
        # Color blindness simulation
        'cvd_protanopia': lambda img: processor.simulate_color_blindness(img, 'protanopia'),
        'cvd_deuteranopia': lambda img: processor.simulate_color_blindness(img, 'deuteranopia'),
        'cvd_tritanopia': lambda img: processor.simulate_color_blindness(img, 'tritanopia'),
        
        # Color blindness correction (Daltonization)
        'cvd_protanopia_corr': lambda img: processor.apply_daltonization(img, 'protanopia'),
        'cvd_deuteranopia_corr': lambda img: processor.apply_daltonization(img, 'deuteranopia'),
        'cvd_tritanopia_corr': lambda img: processor.apply_daltonization(img, 'tritanopia')
    }
    
    # Apply filter if available, otherwise return original image
    if filter_type in filter_mapping:
        return filter_mapping[filter_type](image)
    else:
        print(f"Warning: Unknown filter '{filter_type}', returning original image")
        return image

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'Holo_Process Image Processing',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'features': [
            'Color Blindness Simulator',
            'Daltonization Correction',
            'Advanced Image Filters',
            'Histogram Analysis',
            'Dual Processing Mode'
        ]
    })

@app.route('/info')
def app_info():
    """Application information endpoint"""
    return jsonify({
        'name': 'Holo_Process',
        'version': '2.0.0',
        'author': 'MiniMax Agent',
        'description': 'Advanced Image Processing with Color Blindness Accessibility',
        'features': [
            'Basic Filters (Grayscale, Sepia, Invert)',
            'Brightness/Contrast/Saturation Adjustment',
            'Gamma Correction',
            'Geometric Transformations',
            'Convolution Filters (Blur, Sharpen, Edge Detection)',
            'Morphological Operations',
            'Color Blindness Simulator & Corrector',
            'Histogram Analysis',
            'Real-time Processing'
        ],
        'color_blindness_support': [
            'Protanopia (red-blind) - ~1% of population',
            'Deuteranopia (green-blind) - ~6% of population', 
            'Tritanopia (blue-blind) - ~0.001% of population',
            'Daltonization Correction for all types'
        ],
        'processing_modes': [
            'Client-side (JavaScript)',
            'Server-side (Python/OpenCV)'
        ],
        'supported_formats': ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP'],
        'accessibility_features': [
            'Educational tooltips',
            'Scientific accuracy',
            'Designer testing tools',
            'WCAG compliance assistance'
        ],
        'documentation': 'See README.md for detailed usage instructions'
    })

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🎨 HOLO_PROCESS - ADVANCED IMAGE PROCESSING SERVER")
    print("=" * 80)
    print("📊 Color Blindness Simulator & Corrector enabled")
    print("🔬 Scientific accuracy with research-based algorithms")
    print("🌐 Dual processing mode: Client-side & Server-side")
    print("♿ Accessibility features for inclusive design")
    print("=" * 80)
    print("🚀 Server starting...")
    print("📍 Access points:")
    print("   • Main interface: http://localhost:5000")
    print("   • API health check: http://localhost:5000/health")
    print("   • App information: http://localhost:5000/info")
    print("=" * 80)
    print("✨ Ready to process images with accessibility in mind!")
    print("=" * 80)
    
    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)

# =============================================================================
# END OF SOURCE CODE
# =============================================================================