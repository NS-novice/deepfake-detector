import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ImageProcessor:
    """
    Image processing class for feature extraction from images.
    Extracts various statistical and visual features that may indicate deepfakes.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def extract_features(self, image):
        """
        Extract comprehensive features from an image for deepfake detection
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image for consistent processing (maintain aspect ratio)
        image = self._resize_image(image, max_size=512)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Extract various feature categories
        features = {}
        features.update(self._extract_noise_features(img_array))
        features.update(self._extract_pixel_features(img_array))
        features.update(self._extract_edge_features(img_array))
        features.update(self._extract_texture_features(img_array))
        features.update(self._extract_compression_features(img_array))
        features.update(self._extract_color_features(img_array))
        
        return features
    
    def _resize_image(self, image, max_size=512):
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _extract_noise_features(self, img_array):
        """Extract noise-related features"""
        features = {}
        
        # Convert to grayscale for noise analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur and calculate difference (noise estimation)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # Noise statistics
        features['noise_std'] = np.std(noise) / 255.0
        features['noise_variance'] = np.var(noise) / (255.0 ** 2)
        features['noise_mean'] = abs(np.mean(noise)) / 255.0
        
        # High-frequency noise analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['high_freq_variance'] = np.var(laplacian) / (255.0 ** 2)
        
        return features
    
    def _extract_pixel_features(self, img_array):
        """Extract pixel intensity and distribution features"""
        features = {}
        
        # Overall intensity statistics
        features['mean_intensity'] = np.mean(img_array) / 255.0
        features['intensity_std'] = np.std(img_array) / 255.0
        features['intensity_variance'] = np.var(img_array) / (255.0 ** 2)
        
        # Channel-wise statistics
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = img_array[:, :, i]
            features[f'{channel}_mean'] = np.mean(channel_data) / 255.0
            features[f'{channel}_std'] = np.std(channel_data) / 255.0
        
        # Pixel value distribution
        hist, _ = np.histogram(img_array.flatten(), bins=50, range=(0, 255))
        hist_normalized = hist / np.sum(hist)
        features['pixel_entropy'] = -np.sum(hist_normalized * np.log(hist_normalized + 1e-7))
        
        # Intensity gradient features
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude) / 255.0
        features['gradient_std'] = np.std(gradient_magnitude) / 255.0
        
        return features
    
    def _extract_edge_features(self, img_array):
        """Extract edge-related features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Edge strength distribution
        edge_strength = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        )
        features['edge_strength_mean'] = np.mean(edge_strength) / 255.0
        features['edge_strength_std'] = np.std(edge_strength) / 255.0
        
        # Edge coherence (local edge consistency)
        features['edge_coherence'] = self._calculate_edge_coherence(gray)
        
        return features
    
    def _calculate_edge_coherence(self, gray_img):
        """Calculate edge coherence score"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient angles
        angles = np.arctan2(grad_y, grad_x)
        
        # Calculate local coherence using a sliding window
        coherence_map = np.zeros_like(angles)
        window_size = 5
        
        for i in range(window_size//2, angles.shape[0] - window_size//2):
            for j in range(window_size//2, angles.shape[1] - window_size//2):
                local_angles = angles[i-window_size//2:i+window_size//2+1, 
                                   j-window_size//2:j+window_size//2+1]
                
                # Calculate circular variance
                cos_sum = np.sum(np.cos(2 * local_angles))
                sin_sum = np.sum(np.sin(2 * local_angles))
                coherence_map[i, j] = np.sqrt(cos_sum**2 + sin_sum**2) / local_angles.size
        
        return np.mean(coherence_map)
    
    def _extract_texture_features(self, img_array):
        """Extract texture-related features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern approximation
        features['texture_variance'] = self._calculate_texture_variance(gray)
        
        # Gabor filter responses (simplified)
        features['texture_energy'] = self._calculate_texture_energy(gray)
        
        return features
    
    def _calculate_texture_variance(self, gray_img):
        """Calculate texture variance using local patches"""
        patch_size = 8
        variances = []
        
        for i in range(0, gray_img.shape[0] - patch_size, patch_size):
            for j in range(0, gray_img.shape[1] - patch_size, patch_size):
                patch = gray_img[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
        
        return np.mean(variances) / (255.0 ** 2)
    
    def _calculate_texture_energy(self, gray_img):
        """Calculate texture energy using gradient-based method"""
        # Simple approximation of Gabor filter responses
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        energy = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(energy) / 255.0
    
    def _extract_compression_features(self, img_array):
        """Extract compression-related features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # DCT-based compression analysis (simplified)
        features['compression_consistency'] = self._analyze_compression_consistency(gray)
        
        # Blocking artifacts detection
        features['artifact_score'] = self._detect_blocking_artifacts(gray)
        
        return features
    
    def _analyze_compression_consistency(self, gray_img):
        """Analyze compression consistency across the image"""
        # Divide image into blocks and analyze compression patterns
        block_size = 8
        consistency_scores = []
        
        for i in range(0, gray_img.shape[0] - block_size, block_size):
            for j in range(0, gray_img.shape[1] - block_size, block_size):
                block = gray_img[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # Simple DCT approximation using variance
                block_var = np.var(block)
                consistency_scores.append(block_var)
        
        # Calculate consistency as inverse of variance in block variances
        if len(consistency_scores) > 1:
            consistency = 1.0 / (1.0 + np.var(consistency_scores) / 10000.0)
        else:
            consistency = 1.0
        
        return min(1.0, consistency)
    
    def _detect_blocking_artifacts(self, gray_img):
        """Detect blocking artifacts that might indicate manipulation"""
        # Look for regular patterns that might indicate block-based processing
        h, w = gray_img.shape
        
        # Analyze horizontal and vertical transitions
        h_diffs = []
        v_diffs = []
        
        # Check for regular patterns in differences
        for i in range(8, h-8, 8):
            h_diffs.append(np.mean(np.abs(gray_img[i, :] - gray_img[i-1, :])))
        
        for j in range(8, w-8, 8):
            v_diffs.append(np.mean(np.abs(gray_img[:, j] - gray_img[:, j-1])))
        
        # Calculate artifact score based on regularity
        if len(h_diffs) > 1 and len(v_diffs) > 1:
            h_regularity = np.std(h_diffs) / (np.mean(h_diffs) + 1e-7)
            v_regularity = np.std(v_diffs) / (np.mean(v_diffs) + 1e-7)
            artifact_score = (h_regularity + v_regularity) / 2.0
        else:
            artifact_score = 0.0
        
        return min(1.0, artifact_score)
    
    def _extract_color_features(self, img_array):
        """Extract color-related features"""
        features = {}
        
        # Color distribution entropy
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = img_array[:, :, i]
            hist, _ = np.histogram(channel_data, bins=32, range=(0, 255))
            hist_normalized = hist / (np.sum(hist) + 1e-7)
            features[f'{channel}_entropy'] = -np.sum(hist_normalized * np.log(hist_normalized + 1e-7))
        
        # Overall color entropy
        features['color_entropy'] = np.mean([features['r_entropy'], features['g_entropy'], features['b_entropy']])
        
        # Color variance
        features['color_variance'] = np.var(img_array) / (255.0 ** 2)
        
        # Color channel correlation
        r_flat = img_array[:, :, 0].flatten()
        g_flat = img_array[:, :, 1].flatten()
        b_flat = img_array[:, :, 2].flatten()
        
        features['rg_correlation'] = abs(np.corrcoef(r_flat, g_flat)[0, 1])
        features['rb_correlation'] = abs(np.corrcoef(r_flat, b_flat)[0, 1])
        features['gb_correlation'] = abs(np.corrcoef(g_flat, b_flat)[0, 1])
        
        return features
