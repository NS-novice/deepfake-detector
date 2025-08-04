import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def format_confidence(confidence):
    """Format confidence score as percentage with color coding"""
    percentage = confidence * 100
    if percentage >= 80:
        return f"🔴 {percentage:.1f}% (High Risk)"
    elif percentage >= 50:
        return f"🟡 {percentage:.1f}% (Medium Risk)"
    else:
        return f"🟢 {percentage:.1f}% (Low Risk)"

def create_confidence_chart(feature_scores):
    """Create a bar chart showing individual feature confidence scores"""
    # Prepare data
    features = list(feature_scores.keys())
    scores = [score * 100 for score in feature_scores.values()]
    
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with color coding
    colors = []
    for score in scores:
        if score >= 80:
            colors.append('#ff4444')  # Red for high risk
        elif score >= 50:
            colors.append('#ffaa00')  # Orange for medium risk
        else:
            colors.append('#44aa44')  # Green for low risk
    
    bars = ax.barh(features, scores, color=colors, alpha=0.7)
    
    # Customize the chart
    ax.set_xlabel('Confidence Score (%)')
    ax.set_title('Feature Analysis Breakdown')
    ax.set_xlim(0, 100)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 1, i, f'{score:.1f}%', 
                va='center', fontweight='bold')
    
    # Add reference lines
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Medium threshold')
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='High threshold')
    
    # Format feature names for better readability
    formatted_features = [feature.replace('_', ' ').title() for feature in features]
    ax.set_yticklabels(formatted_features)
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_comparison_visualization(original_features, processed_features=None):
    """Create a comparison visualization between original and processed features"""
    if processed_features is None:
        # Single image analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature distribution
        features = list(original_features.keys())
        values = list(original_features.values())
        
        ax1.bar(range(len(features)), values, alpha=0.7)
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Values')
        ax1.set_title('Feature Values Distribution')
        ax1.set_xticks(range(len(features)))
        ax1.set_xticklabels(features, rotation=45, ha='right')
        
        # Feature importance (mock visualization)
        importance_scores = np.random.rand(len(features))  # Placeholder
        ax2.pie(importance_scores, labels=features, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Importance (Relative)')
        
    else:
        # Comparison between two images
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = list(original_features.keys())
        orig_values = [original_features[f] for f in features]
        proc_values = [processed_features.get(f, 0) for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax.bar(x - width/2, orig_values, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, proc_values, width, label='Comparison', alpha=0.7)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title('Feature Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    return fig

def get_risk_assessment(confidence_score):
    """Get risk assessment based on confidence score"""
    if confidence_score >= 0.8:
        return {
            'level': 'HIGH',
            'color': 'red',
            'emoji': '🚨',
            'description': 'Strong indicators of manipulation detected. This image likely contains deepfake characteristics.',
            'recommendations': [
                'Exercise extreme caution before trusting this image',
                'Consider additional verification methods',
                'Check the source and context carefully'
            ]
        }
    elif confidence_score >= 0.5:
        return {
            'level': 'MEDIUM',
            'color': 'orange', 
            'emoji': '⚠️',
            'description': 'Some suspicious patterns detected. The image may have been manipulated.',
            'recommendations': [
                'Verify the source of the image',
                'Look for additional evidence of authenticity',
                'Consider the context and plausibility'
            ]
        }
    else:
        return {
            'level': 'LOW',
            'color': 'green',
            'emoji': '✅',
            'description': 'The image appears to be authentic with no strong indicators of manipulation.',
            'recommendations': [
                'Image appears legitimate based on analysis',
                'Standard verification practices still apply',
                'Consider the source and context as always'
            ]
        }

def display_technical_details(features, detection_result):
    """Display detailed technical information about the analysis"""
    st.subheader("🔬 Technical Analysis Details")
    
    with st.expander("Raw Feature Values"):
        st.json(features)
    
    with st.expander("Detection Algorithm Details"):
        st.write("**Algorithm Components:**")
        st.write("- **Anomaly Detection:** Isolation Forest")
        st.write("- **Feature Extraction:** Multi-domain analysis")
        st.write("- **Confidence Calculation:** Weighted ensemble")
        
        st.write("**Feature Categories:**")
        st.write("- Noise Analysis: Statistical noise properties")
        st.write("- Pixel Features: Intensity distributions and gradients") 
        st.write("- Edge Features: Edge density and coherence")
        st.write("- Texture Features: Local texture patterns")
        st.write("- Compression Features: Artifact detection")
        st.write("- Color Features: Color distribution analysis")
    
    with st.expander("Scoring Methodology"):
        st.write("**Confidence Score Calculation:**")
        st.write("1. Individual feature scores (0-1 scale)")
        st.write("2. Weighted combination based on feature importance")
        st.write("3. Anomaly detection score integration")
        st.write("4. Final normalization and calibration")
        
        st.write("**Feature Weights:**")
        weights = {
            'Noise Patterns': '25%',
            'Pixel Anomalies': '20%', 
            'Edge Inconsistencies': '20%',
            'Compression Artifacts': '20%',
            'Color Distribution': '15%'
        }
        for feature, weight in weights.items():
            st.write(f"- {feature}: {weight}")

def save_analysis_report(image_name, detection_result, features):
    """Generate a downloadable analysis report"""
    report = f"""
DEEPFAKE DETECTION ANALYSIS REPORT
==================================

Image: {image_name}
Analysis Date: {np.datetime64('now')}

DETECTION RESULTS:
- Prediction: {'POTENTIAL DEEPFAKE' if detection_result['prediction'] else 'APPEARS AUTHENTIC'}
- Confidence: {detection_result['confidence']*100:.1f}%
- Anomaly Score: {detection_result['anomaly_score']:.3f}

FEATURE ANALYSIS:
"""
    
    for feature, score in detection_result['feature_analysis'].items():
        report += f"- {feature.replace('_', ' ').title()}: {score*100:.1f}%\n"
    
    report += f"""

RAW FEATURES:
{str(features)}

METHODOLOGY:
This analysis uses statistical and machine learning techniques to identify
potential deepfake characteristics in images. The algorithm examines multiple
feature categories including noise patterns, pixel distributions, edge 
consistency, texture properties, compression artifacts, and color distributions.

DISCLAIMER:
This tool is for educational and research purposes. Results should not be
considered definitive proof of image authenticity or manipulation. Always
verify through multiple sources and methods when authenticity is critical.
"""
    
    return report
