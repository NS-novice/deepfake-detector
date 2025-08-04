import streamlit as st
from PIL import Image
import time
from typing import Optional, Dict, Any
import io
import tempfile
import os
from contextlib import contextmanager

# Import Reality Defender SDK
try:
    from realitydefender import RealityDefender
except ImportError:
    st.error("❌ Reality Defender SDK not installed. Please install it with: pip install realitydefender")
    st.stop()

# ───────────────────────── CONFIG
try:
    API_KEY = st.secrets["REALITY_DEFENDER_API_KEY"]
except KeyError:
    st.error("❌ API key not found. Please add REALITY_DEFENDER_API_KEY to your Streamlit secrets.")
    st.stop()

# Initialize Reality Defender SDK
try:
    rd = RealityDefender(api_key=API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize Reality Defender SDK: {e}")
    st.stop()

# File size limit (15MB)
MAX_FILE_SIZE = 15 * 1024 * 1024

# ───────────────────────── HELPER FUNCTIONS
@contextmanager
def temp_file_context(file_bytes, suffix='.jpg'):
    """Context manager for temporary file handling"""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        yield temp_file_path
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def discover_sdk_methods() -> list:
    """Discover available SDK methods"""
    available_methods = []
    for attr_name in dir(rd):
        if not attr_name.startswith('_'):
            attr = getattr(rd, attr_name)
            if callable(attr):
                available_methods.append(attr_name)
    return available_methods

def extract_job_id(response):
    """Extract job ID from various response formats"""
    if isinstance(response, str):
        return response
    elif isinstance(response, dict):
        # Try common field names
        for field in ['id', 'job_id', 'request_id', 'upload_id', 'task_id', 'uuid']:
            if field in response:
                return response[field]
    return None

def poll_for_results(rd, job_id, max_attempts=20):
    """Poll for results with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            if hasattr(rd, 'get_result'):
                result = rd.get_result(job_id)
            elif hasattr(rd, 'get_prediction'):
                result = rd.get_prediction(job_id)
            elif hasattr(rd, 'check_status'):
                result = rd.check_status(job_id)
            else:
                raise Exception("No polling method available")

            if result and isinstance(result, dict):
                status = str(result.get('status', '')).lower()
                if status in ['completed', 'done', 'finished', 'success']:
                    return result
                elif status in ['failed', 'error']:
                    raise Exception(f"Analysis failed: {result}")
                elif status in ['processing', 'pending', 'running']:
                    # Continue polling
                    pass
                else:
                    # Check if we have actual results regardless of status
                    if any(key in result for key in ['prediction', 'score', 'label', 'confidence']):
                        return result

            # Exponential backoff: 1s, 2s, 4s, 8s, max 10s
            wait_time = min(2 ** attempt, 10)
            time.sleep(wait_time)

        except Exception as e:
            if attempt == max_attempts - 1:
                raise Exception(f"Polling failed after {max_attempts} attempts: {e}")
            time.sleep(2)

    raise TimeoutError("Polling timed out")

def validate_file(uploaded_file) -> bool:
    """Validate uploaded file size and type"""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum size is 15MB.")
        return False
    return True

def analyze_with_sdk(file_bytes: bytes, filename: str = "image.jpg") -> Optional[Dict[Any, Any]]:
    """Analyze image using Reality Defender SDK with method discovery"""

    available_methods = discover_sdk_methods()
    st.info(f"🔍 Available SDK methods: {', '.join(available_methods)}")

    result = None

    with st.spinner("🔄 Analyzing with Reality Defender SDK..."):

        # Method 1: Try direct upload methods that might return immediate results
        direct_methods = [
            ('upload_sync', lambda: rd.upload_sync(file_bytes)),
            ('analyze_image', lambda: rd.analyze_image(file_bytes)),
            ('detect_image', lambda: rd.detect_image(file_bytes)),
            ('predict', lambda: rd.predict(file_bytes)),
            ('detect', lambda: rd.detect(file_bytes)),
            ('classify', lambda: rd.classify(file_bytes)),
        ]

        for method_name, method_call in direct_methods:
            if method_name in available_methods:
                try:
                    st.info(f"📤 Trying {method_name}...")
                    result = method_call()
                    if result:
                        st.success(f"✅ {method_name} succeeded!")
                        break
                except Exception as e:
                    st.warning(f"⚠️ {method_name} failed: {e}")
                    continue

        # Method 2: Try file-based methods if available
        if not result and 'detect_file' in available_methods:
            try:
                st.info("📁 Trying detect_file with temporary file...")
                with temp_file_context(file_bytes) as temp_path:
                    result = rd.detect_file(temp_path)
                if result:
                    st.success("✅ detect_file succeeded!")
            except Exception as e:
                st.warning(f"⚠️ detect_file failed: {e}")

        # Method 3: Try async upload + polling pattern
        if not result and 'upload' in available_methods:
            try:
                st.info("📤 Trying async upload + polling...")

                # Try upload with just bytes
                upload_response = rd.upload(file_bytes)
                st.info(f"Upload response: {upload_response}")

                job_id = extract_job_id(upload_response)

                if job_id:
                    st.info(f"🔄 Polling for results (ID: {job_id})...")
                    result = poll_for_results(rd, job_id)
                    st.success("✅ Async upload + polling succeeded!")
                else:
                    st.warning("❌ No job ID returned from upload")

            except Exception as e:
                st.warning(f"⚠️ Async upload failed: {e}")

        # Method 4: Try upload with file-like object
        if not result and 'upload' in available_methods:
            try:
                st.info("📄 Trying upload with file-like object...")
                file_obj = io.BytesIO(file_bytes)
                file_obj.name = filename
                result = rd.upload(file_obj)
                if result:
                    st.success("✅ Upload with file object succeeded!")
            except Exception as e:
                st.warning(f"⚠️ Upload with file object failed: {e}")

    if result:
        # Debug: Show raw SDK response
        with st.expander("🔍 Debug: Raw SDK Response"):
            st.json(result)
        return result
    else:
        st.error("❌ All SDK methods failed")
        with st.expander("🛠️ Troubleshooting"):
            st.write("**Available methods:**", available_methods)
            st.write("**Try checking the Reality Defender documentation for correct usage patterns.**")
        return None

def mock_analysis() -> Dict[Any, Any]:
    """Mock analysis for demo purposes"""
    import random

    # Simulate processing time
    time.sleep(2)

    # Generate mock results
    is_fake = random.choice([True, False])
    confidence = random.uniform(0.7, 0.95) if is_fake else random.uniform(0.6, 0.9)

    return {
        "label": "fake" if is_fake else "real",
        "score": confidence,
        "confidence": confidence,
        "model": "demo-mode",
        "note": "This is a mock result for demonstration purposes"
    }

def display_results(result: Dict[Any, Any]) -> None:
    """Display analysis results in a user-friendly format"""

    # Extract score/confidence
    score = (result.get("score") or 
             result.get("confidence") or 
             result.get("probability"))

    # Extract label/prediction
    label = (result.get("label") or 
             result.get("prediction") or 
             result.get("classification"))

    # Handle boolean fields
    if "is_fake" in result:
        label = "fake" if result["is_fake"] else "real"
    elif "is_deepfake" in result:
        label = "fake" if result["is_deepfake"] else "real"

    # Handle numeric predictions (threshold-based)
    if label is None and score is not None:
        threshold = 0.5
        label = "fake" if score > threshold else "real"

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        if label:
            st.metric("🎯 Result", str(label).title())
        else:
            st.metric("🎯 Result", "Unknown")

    with col2:
        if score is not None:
            # Convert to percentage if needed
            if score > 1:
                score = score / 100
            st.metric("📊 Confidence", f"{score:.1%}")
        else:
            st.metric("📊 Confidence", "N/A")

    # Color-coded result
    if str(label).lower() == "fake":
        st.error("⚠️ **Potential deepfake/manipulation detected**")
    elif str(label).lower() == "real":
        st.success("✅ **Likely authentic image**")
    else:
        st.info(f"ℹ️ **Result: {label}**")

    # Demo mode warning
    if result.get("model") == "demo-mode":
        st.warning("🎭 **DEMO MODE**: These are mock results for demonstration only!")

    # Detailed results
    with st.expander("📋 View Detailed Results"):
        st.json(result)

# ───────────────────────── STREAMLIT UI
st.set_page_config(
    page_title="Reality Defender - Deepfake Detection", 
    page_icon="🕵️",
    layout="centered"
)

# Initialize session state
if 'file_bytes' not in st.session_state:
    st.session_state.file_bytes = None
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None

st.title("🕵️‍♀️ Reality Defender — Deepfake Detection")
st.caption("Powered by Reality Defender Python SDK | v3.0 - Auto-discovery methods")

# Instructions
with st.expander("📖 How to use"):
    st.write("""
    1. **Upload an image** (JPG, JPEG, or PNG format)
    2. **Wait for analysis** - the app will automatically try different SDK methods
    3. **Review results** and confidence score

    **Supported formats:** JPG, JPEG, PNG  
    **Maximum file size:** 15 MB

    **Note:** This app automatically discovers and tries the correct SDK methods for your version.
    """)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Demo mode toggle
    demo_mode = st.checkbox(
        "🎭 Demo Mode", 
        help="Use mock results for testing the interface"
    )

    if demo_mode:
        st.warning("🎭 Demo mode enabled - results are not real!")

    # SDK status
    st.subheader("📡 SDK Status")
    try:
        available_methods = discover_sdk_methods()
        st.success(f"✅ SDK Ready ({len(available_methods)} methods)")

        with st.expander("Available methods"):
            for method in available_methods:
                st.write(f"• `{method}()`")

    except Exception as e:
        st.error(f"❌ SDK Error: {e}")

# Main upload interface
uploaded_file = st.file_uploader(
    "📤 Upload an image for deepfake detection",
    type=["jpg", "jpeg", "png"],
    help="Maximum file size: 15MB"
)

if uploaded_file is not None:
    # Validate file
    if not validate_file(uploaded_file):
        st.stop()

    # Cache file bytes to avoid re-reading on rerun
    if (st.session_state.file_bytes is None or 
        st.session_state.last_file_name != uploaded_file.name):

        st.session_state.file_bytes = uploaded_file.read()
        st.session_state.last_file_name = uploaded_file.name

    # Display uploaded image
    try:
        image = Image.open(io.BytesIO(st.session_state.file_bytes))
        st.image(image, caption=f'📷 Uploaded: {uploaded_file.name}', use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not display image: {e}")
        st.stop()

    # Analysis section
    st.markdown("---")

    if demo_mode:
        # Demo analysis
        st.info("🎭 Running demo analysis...")
        with st.spinner("Generating mock results..."):
            result = mock_analysis()

        st.success("🎭 Demo analysis complete!")
        display_results(result)

    else:
        # Real SDK analysis
        st.info("🔍 Starting Reality Defender analysis...")

        result = analyze_with_sdk(st.session_state.file_bytes, uploaded_file.name)

        if result:
            st.success("✅ Analysis complete!")
            display_results(result)
        else:
            st.error("❌ Analysis failed. Try Demo Mode to test the interface.")

            with st.expander("💡 Troubleshooting Tips"):
                st.write("""
                1. **Check your API key** - Make sure it's valid and has sufficient credits
                2. **Try a different image** - Some formats might not be supported
                3. **Check the SDK documentation** - Method names might have changed
                4. **Use Demo Mode** - To test the interface while debugging
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
<b>Reality Defender SDK Integration</b><br>
This app automatically discovers and uses the correct SDK methods for your version.<br>
<i>Keep your API key secure in Streamlit secrets.</i>
</div>
""", unsafe_allow_html=True)
