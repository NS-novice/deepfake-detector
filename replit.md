# Deepfake Detection Tool

## Overview

This is a Streamlit-based web application for detecting potentially AI-generated or manipulated images (deepfakes). The system uses statistical analysis and machine learning techniques to analyze various image characteristics and provide confidence scores indicating the likelihood that an image has been artificially generated or manipulated.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular Python architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for user interaction
- **Core Detection**: Machine learning-based analysis using statistical features
- **Image Processing**: Feature extraction pipeline for analyzing image characteristics
- **Utilities**: Helper functions for visualization and formatting

## Key Components

### 1. Streamlit Frontend (app.py)
- **Purpose**: Provides the web interface for users to upload images and view results
- **Key Features**: 
  - Image upload functionality
  - Real-time analysis display
  - Confidence score visualization
  - Educational information sidebar
- **Caching Strategy**: Uses `@st.cache_resource` for efficient model loading

### 2. Deepfake Detector (deepfake_detector.py)
- **Purpose**: Core ML component for deepfake detection
- **Algorithm**: Uses Isolation Forest for anomaly detection combined with statistical baseline comparison
- **Training Approach**: Self-supervised learning with synthetic baseline features
- **Key Features**:
  - Lightweight CPU-based processing
  - Statistical pattern recognition
  - Anomaly scoring system

### 3. Image Processor (image_processor.py)
- **Purpose**: Extracts multiple feature categories from input images
- **Feature Categories**:
  - Noise analysis patterns
  - Pixel intensity distributions
  - Edge detection characteristics
  - Texture analysis
  - Compression artifact detection
  - Color distribution analysis
- **Preprocessing**: Automatic image resizing and format normalization

### 4. Utilities (utils.py)
- **Purpose**: Visualization and formatting helpers
- **Components**:
  - Confidence score formatting with risk levels
  - Interactive chart generation for feature breakdown
  - Color-coded risk assessment display

## Data Flow

1. **Image Upload**: User uploads image through Streamlit interface
2. **Preprocessing**: Image is converted to RGB and resized for consistent processing
3. **Feature Extraction**: Multiple statistical and visual features are extracted
4. **Analysis**: Features are compared against baseline patterns using ML models
5. **Scoring**: Individual feature scores are combined into overall confidence
6. **Visualization**: Results are displayed with charts and risk assessment

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Isolation Forest, preprocessing, and similarity metrics
- **numpy**: Numerical computations and array operations
- **scipy**: Advanced image processing functions

### Image Processing
- **PIL (Pillow)**: Image loading, conversion, and basic processing
- **opencv-python (cv2)**: Advanced computer vision operations

### Web Framework
- **streamlit**: Web application framework and UI components
- **matplotlib**: Chart generation and data visualization

## Deployment Strategy

### Current Setup
- **Runtime**: CPU-based processing suitable for lightweight deployment
- **Architecture**: Single-container Streamlit application
- **Resource Requirements**: Minimal - designed for educational/demonstration use

### Scalability Considerations
- Model loading is cached to improve performance
- Image processing is optimized for speed over accuracy
- Memory-efficient feature extraction pipeline
- No persistent storage requirements (stateless design)

### Potential Enhancements
- GPU acceleration for more sophisticated deep learning models
- Database integration for storing analysis results
- API endpoints for programmatic access
- Enhanced model training with real datasets
- Integration with cloud storage for large file handling

## Technical Notes

- The detection system is designed for educational purposes and should not be considered production-ready for security applications
- Uses synthetic baseline training data rather than real deepfake datasets
- Focuses on statistical analysis rather than deep neural networks for broader compatibility
- All processing happens locally without external API dependencies