# Holo_Process - Advanced Image Processing with Color Blindness Accessibility

A comprehensive web-based image processing application featuring client-side processing, server-side capabilities, and advanced color blindness simulation and correction tools.

## 🎨 Features

### Core Image Processing
- **Basic Filters**: Grayscale, Sepia, Invert
- **Color Adjustments**: Brightness, Contrast, Saturation
- **Gamma Correction**: Multiple gamma values for tone mapping
- **Geometric Transforms**: Rotation, Translation, Shearing
- **Convolution Filters**: Blur, Sharpen, Edge Detection, Emboss
- **Morphological Operations**: Erosion, Dilation
- **Advanced Effects**: Histogram Equalization, Cartoon Filter, Noise Addition

### 🎯 Color Blindness Accessibility (NEW)
- **Color Vision Deficiency Simulation**:
  - Protanopia (red-blind) - ~1% of population
  - Deuteranopia (green-blind) - ~6% of population  
  - Tritanopia (blue-blind) - ~0.001% of population
- **Daltonization Correction**: Enhances lost color information for accessibility
- **Educational Tooltips**: Explains each condition with scientific accuracy
- **Designer Testing**: Helps create accessible charts, maps, and UI elements

### Technical Features
- **Dual Processing Mode**: Client-side (JavaScript) and Server-side (Python)
- **Real-time Histograms**: Grayscale and RGB channel analysis
- **Responsive Design**: Works on desktop and mobile
- **Drag & Drop Upload**: Intuitive image loading
- **Download Support**: Save processed images locally or via server

## 🚀 Quick Start

### Option 1: Client-Side Only (Static)
1. Open `index.html` in any modern web browser
2. Upload an image and apply filters instantly
3. No server setup required

### Option 2: Full Stack (Recommended)
1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask Server**:
   ```bash
   python app.py
   ```

3. **Access the Application**:
   - Open browser to `http://localhost:5000`
   - Or use the enhanced version: `index_enhanced.html`

4. **Toggle Server Mode**:
   - Click "Toggle Server Mode" in the header
   - Status indicator shows current mode (Client/Server)

## 📁 Project Structure

```
├── app.py                    # Flask backend with OpenCV processing
├── index.html               # Original client-side version
├── index_enhanced.html      # Enhanced version with server integration
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
└── user_input_files/       # Input files directory
    └── index.html          # Your original HTML file
```

## 🔧 Technical Implementation

### Color Blindness Simulation
Uses scientifically accurate linear RGB transformation matrices:

- **Protanopia Matrix**:
  ```
  [0.567, 0.433, 0.000]
  [0.558, 0.442, 0.000] 
  [0.000, 0.242, 0.758]
  ```

- **Deuteranopia Matrix**:
  ```
  [0.625, 0.375, 0.000]
  [0.700, 0.300, 0.000]
  [0.000, 0.300, 0.700]
  ```

- **Tritanopia Matrix**:
  ```
  [0.950, 0.050, 0.000]
  [0.000, 0.433, 0.567]
  [0.000, 0.475, 0.525]
  ```

### Daltonization Algorithm
1. Simulate color vision deficiency on image copy
2. Calculate error: `original - simulated`
3. Shift lost color information to visible channels
4. For red-green types: boost into blue channel
5. For blue type: boost into red and green channels

### Server Integration
- **REST API Endpoints**:
  - `/process` - Apply image filters
  - `/histogram` - Calculate image histograms
  - `/download` - Download processed images
  - `/health` - Server health check
  - `/info` - Application information

## 🎓 Educational Value

### For Designers
- Test website/app accessibility before deployment
- Understand how color-blind users experience content
- Design inclusive color schemes for charts and data visualization
- Comply with WCAG accessibility guidelines

### For Developers
- Learn image processing algorithms
- Understand color science and human vision
- Practice web development with Canvas API
- Implement accessibility features

### For Researchers
- Study color vision deficiency patterns
- Analyze image processing techniques
- Research accessibility technology
- Educational demonstrations

## 🌐 Browser Compatibility

- **Modern Browsers**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **Required Features**: Canvas API, File API, Fetch API
- **Mobile Support**: iOS Safari 13+, Chrome Mobile 80+

## 📊 Performance

- **Client Processing**: Instant for small images (<2MB)
- **Server Processing**: Better for large images and complex filters
- **Memory Usage**: Efficient with canvas-based processing
- **Responsiveness**: Non-blocking UI with async operations

## 🔒 Privacy & Security

- **Client-Side Mode**: All processing happens in browser, no data leaves device
- **Server Mode**: Images processed temporarily, not stored permanently
- **No Tracking**: No analytics or user data collection
- **Open Source**: All code available for review

## 🤝 Contributing

This project demonstrates advanced web development techniques:
- Canvas-based image manipulation
- Scientific color vision modeling
- Accessibility-first design
- Dual-mode architecture (client/server)
- Real-time histogram analysis

## 📚 References

- Color blindness prevalence: ~8% of males, ~0.5% of females
- Simulation matrices: Based on academic research (Coblis/Vischeck algorithms)
- Daltonization: Adapted from established accessibility research
- WCAG Guidelines: Web Content Accessibility Guidelines 2.1

---

