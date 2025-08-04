import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class DeepfakeDetector:
    """
    A deepfake detection system using statistical analysis and anomaly detection.
    Uses lightweight ML techniques suitable for CPU processing.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.is_trained = False
        self._initialize_baseline()
    
    def _initialize_baseline(self):
        """Initialize baseline patterns for comparison"""
        # Create synthetic baseline features representing typical real image characteristics
        np.random.seed(42)
        
        # Generate baseline feature patterns for real images
        self.real_image_baseline = {
            'noise_std_range': (0.1, 0.3),
            'pixel_intensity_mean': (0.3, 0.7),
            'edge_density_range': (0.15, 0.45),
            'compression_consistency': (0.7, 0.95),
            'color_distribution_entropy': (6.0, 8.5)
        }
        
        # Create training data for anomaly detection
        baseline_features = []
        for _ in range(1000):
            features = self._generate_baseline_features()
            baseline_features.append(features)
        
        baseline_features = np.array(baseline_features)
        self.scaler.fit(baseline_features)
        self.anomaly_detector.fit(self.scaler.transform(baseline_features))
        self.is_trained = True
    
    def _generate_baseline_features(self):
        """Generate synthetic baseline features for training"""
        features = []
        
        # Noise characteristics
        features.append(np.random.uniform(0.1, 0.3))  # noise_std
        features.append(np.random.uniform(0.05, 0.15))  # noise_variance
        
        # Pixel intensity features
        features.append(np.random.uniform(0.3, 0.7))  # mean_intensity
        features.append(np.random.uniform(0.15, 0.35))  # intensity_std
        
        # Edge and texture features
        features.append(np.random.uniform(0.15, 0.45))  # edge_density
        features.append(np.random.uniform(0.1, 0.3))  # texture_variance
        
        # Compression and artifacts
        features.append(np.random.uniform(0.7, 0.95))  # compression_consistency
        features.append(np.random.uniform(0.05, 0.2))  # artifact_score
        
        # Color distribution
        features.append(np.random.uniform(6.0, 8.5))  # color_entropy
        features.append(np.random.uniform(0.2, 0.6))  # color_variance
        
        return features
    
    def predict(self, features):
        """
        Predict whether an image is a deepfake based on extracted features
        
        Args:
            features (dict): Dictionary of extracted image features
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if not self.is_trained:
            raise RuntimeError("Detector not initialized properly")
        
        # Convert features to array format
        feature_array = self._features_to_array(features)
        
        # Normalize features
        normalized_features = self.scaler.transform([feature_array])
        
        # Get anomaly score
        anomaly_score = self.anomaly_detector.decision_function(normalized_features)[0]
        
        # Analyze individual features
        feature_analysis = self._analyze_features(features)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(feature_analysis, anomaly_score)
        
        # Make prediction
        is_deepfake = overall_confidence > 0.5
        
        return {
            'prediction': is_deepfake,
            'confidence': overall_confidence,
            'anomaly_score': anomaly_score,
            'feature_analysis': feature_analysis
        }
    
    def _features_to_array(self, features):
        """Convert feature dictionary to array format"""
        return [
            features.get('noise_std', 0),
            features.get('noise_variance', 0),
            features.get('mean_intensity', 0),
            features.get('intensity_std', 0),
            features.get('edge_density', 0),
            features.get('texture_variance', 0),
            features.get('compression_consistency', 0),
            features.get('artifact_score', 0),
            features.get('color_entropy', 0),
            features.get('color_variance', 0)
        ]
    
    def _analyze_features(self, features):
        """Analyze individual features for deepfake indicators"""
        analysis = {}
        
        # Noise analysis
        noise_std = features.get('noise_std', 0)
        noise_baseline = self.real_image_baseline['noise_std_range']
        if noise_std < noise_baseline[0] or noise_std > noise_baseline[1]:
            analysis['noise_patterns'] = min(1.0, abs(noise_std - np.mean(noise_baseline)) * 2)
        else:
            analysis['noise_patterns'] = 0.1
        
        # Pixel intensity analysis
        intensity_mean = features.get('mean_intensity', 0)
        intensity_baseline = self.real_image_baseline['pixel_intensity_mean']
        if intensity_mean < intensity_baseline[0] or intensity_mean > intensity_baseline[1]:
            analysis['pixel_anomalies'] = min(1.0, abs(intensity_mean - np.mean(intensity_baseline)) * 1.5)
        else:
            analysis['pixel_anomalies'] = 0.1
        
        # Edge density analysis
        edge_density = features.get('edge_density', 0)
        edge_baseline = self.real_image_baseline['edge_density_range']
        if edge_density < edge_baseline[0] or edge_density > edge_baseline[1]:
            analysis['edge_inconsistencies'] = min(1.0, abs(edge_density - np.mean(edge_baseline)) * 2)
        else:
            analysis['edge_inconsistencies'] = 0.1
        
        # Compression analysis
        compression = features.get('compression_consistency', 0)
        compression_baseline = self.real_image_baseline['compression_consistency']
        if compression < compression_baseline[0]:
            analysis['compression_artifacts'] = min(1.0, (compression_baseline[0] - compression) * 2)
        else:
            analysis['compression_artifacts'] = 0.1
        
        # Color distribution analysis
        color_entropy = features.get('color_entropy', 0)
        entropy_baseline = self.real_image_baseline['color_distribution_entropy']
        if color_entropy < entropy_baseline[0] or color_entropy > entropy_baseline[1]:
            analysis['color_distribution'] = min(1.0, abs(color_entropy - np.mean(entropy_baseline)) / 2)
        else:
            analysis['color_distribution'] = 0.1
        
        return analysis
    
    def _calculate_confidence(self, feature_analysis, anomaly_score):
        """Calculate overall confidence score"""
        # Weight the feature scores
        weights = {
            'noise_patterns': 0.25,
            'pixel_anomalies': 0.20,
            'edge_inconsistencies': 0.20,
            'compression_artifacts': 0.20,
            'color_distribution': 0.15
        }
        
        # Calculate weighted average of feature scores
        weighted_score = sum(
            feature_analysis.get(feature, 0) * weight
            for feature, weight in weights.items()
        )
        
        # Incorporate anomaly score (normalize from [-1, 1] to [0, 1])
        normalized_anomaly = max(0, min(1, (-anomaly_score + 1) / 2))
        
        # Combine scores
        final_confidence = (weighted_score * 0.7) + (normalized_anomaly * 0.3)
        
        return min(1.0, max(0.0, final_confidence))
