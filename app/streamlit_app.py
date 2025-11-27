"""
Smart Parking Detection System - Streamlit Demo App
----------------------------------------------------
Interactive web interface for parking space detection.

Supports two detection backends:
1. Local YOLOv8 (requires ultralytics + GPU recommended)
2. Roboflow API (cloud-based, no local GPU needed)

Run with:
    streamlit run app/streamlit_app.py

Or from project root:
    streamlit run app/streamlit_app.py --server.port 8501
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import tempfile
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yaml

from utils.detector import Detection
from utils.slot_matcher import SlotMatcher, SlotStatus, create_grid_slots
from utils.visualization import ParkingVisualizer
from utils.auto_slot_detector import AutoSlotDetector


# Try to import both detector backends
YOLO_AVAILABLE = False
ROBOFLOW_AVAILABLE = False

try:
    from utils.detector import ParkingDetector
    YOLO_AVAILABLE = True
except ImportError:
    ParkingDetector = None

try:
    from utils.roboflow_detector import RoboflowDetector
    ROBOFLOW_AVAILABLE = True
except ImportError:
    RoboflowDetector = None


# Page configuration
st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_matcher(config_path: str):
    """Load slot matcher with caching."""
    return SlotMatcher(config_path=config_path)


def create_default_slots():
    """Create default parking slot configuration."""
    return create_grid_slots(
        start_x=100, start_y=100,
        slot_width=80, slot_height=100,
        rows=4, cols=6,
        h_spacing=10, v_spacing=20
    )


def main():
    # Title and description
    st.title("🅿️ Smart Parking Detection System")
    st.markdown("""
    Upload **any** parking lot image and get instant parking space detection!

    🤖 **Auto-Detect Mode**: AI automatically finds parking spaces in your image
    📁 **Upload Config**: Use saved configurations for recurring lots
    🎛️ **Manual Grid**: Customize grid parameters yourself
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Detection backend selection
    st.sidebar.subheader("Detection Backend")
    
    backend_options = []
    if ROBOFLOW_AVAILABLE:
        backend_options.append("☁️ Roboflow API (Cloud)")
    if YOLO_AVAILABLE:
        backend_options.append("💻 Local YOLOv8")
    
    if not backend_options:
        st.error("No detection backend available. Please install ultralytics or requests.")
        st.stop()
    
    selected_backend = st.sidebar.radio(
        "Choose detection backend:",
        options=backend_options,
        help="Roboflow: No GPU needed, uses cloud API. Local: Requires ultralytics package."
    )
    
    use_roboflow = "Roboflow" in selected_backend
    
    # Roboflow settings
    if use_roboflow:
        st.sidebar.subheader("Roboflow Settings")
        
        # API key input (with default from your guide)
        roboflow_api_key = st.sidebar.text_input(
            "API Key",
            value=os.getenv("ROBOFLOW_API_KEY", "i6ssN6FE5PzINBYzJxHN"),
            type="password",
            help="Your Roboflow API key"
        )
        
        # Show workspace info
        st.sidebar.caption("Workspace: `parkpark-zclps`")
        st.sidebar.caption("Workflow: `find-cars-2`")
        
        confidence_threshold = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05
        )
    
    else:
        # Local YOLOv8 settings
        st.sidebar.subheader("Model Settings")
        
        model_options = {
            "YOLOv8 Nano (Fastest)": "yolov8n.pt",
            "YOLOv8 Small": "yolov8s.pt",
            "YOLOv8 Medium (Recommended)": "yolov8m.pt",
            "YOLOv8 Large": "yolov8l.pt",
            "YOLOv8 XLarge (Most Accurate)": "yolov8x.pt",
        }
        
        model_choice = st.sidebar.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=2,  # Default to Medium
            help="Larger models are more accurate but slower"
        )
        model_path = model_options[model_choice]
        
        # Custom model upload
        custom_model = st.sidebar.file_uploader(
            "Or upload custom model (.pt)",
            type=['pt'],
            help="Upload your fine-tuned YOLOv8 model"
        )
        
        if custom_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
                f.write(custom_model.read())
                model_path = f.name
            st.sidebar.success(f"Using custom model: {custom_model.name}")
        
        confidence_threshold = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence for vehicle detection"
        )
        
        device = st.sidebar.selectbox(
            "Device",
            options=["auto", "cpu", "cuda"],
            index=0,
            help="Device for inference"
        )
    
    # Common settings
    st.sidebar.subheader("Slot Matching")
    
    iou_threshold = st.sidebar.slider(
        "Slot IoU Threshold",
        min_value=0.1,
        max_value=0.7,
        value=0.3,
        step=0.05,
        help="Minimum overlap to consider a slot occupied"
    )
    
    # Slot configuration
    st.sidebar.subheader("Parking Slot Config")

    slot_mode = st.sidebar.radio(
        "Slot Detection Mode:",
        options=["🤖 Auto-Detect (Smart)", "📁 Upload Config", "🎛️ Manual Grid"],
        index=0,
        help="Auto-Detect: AI analyzes your image | Upload: Use saved config | Manual: Set grid manually"
    )

    config_file = None
    use_auto_detect = slot_mode == "🤖 Auto-Detect (Smart)"
    use_manual_grid = slot_mode == "🎛️ Manual Grid"

    if slot_mode == "📁 Upload Config":
        config_file = st.sidebar.file_uploader(
            "Upload slot config (YAML)",
            type=['yaml', 'yml'],
            help="Upload a parking lot configuration file"
        )

    if use_manual_grid:
        st.sidebar.markdown("**Grid Parameters:**")
        manual_rows = st.sidebar.slider("Rows", 2, 10, 3)
        manual_cols = st.sidebar.slider("Columns", 2, 15, 6)
    
    # Display options
    st.sidebar.subheader("Display Options")
    show_detections = st.sidebar.checkbox("Show detection boxes", value=True)
    show_slot_ids = st.sidebar.checkbox("Show slot IDs", value=True)
    show_schematic = st.sidebar.checkbox("Show schematic map", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Input Image")
        
        # Demo images section
        demo_dir = Path(__file__).parent.parent / "demo_images"
        demo_images = list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png")) if demo_dir.exists() else []
        
        input_method = st.radio(
            "Choose input method:",
            options=["Upload Image", "Use Demo Image"],
            horizontal=True
        )
        
        uploaded_image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload parking lot image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a top-down view of a parking lot"
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file)
        
        elif input_method == "Use Demo Image":
            if demo_images:
                selected_demo = st.selectbox(
                    "Select demo image:",
                    options=[p.name for p in demo_images]
                )
                demo_path = demo_dir / selected_demo
                uploaded_image = Image.open(demo_path)
            else:
                st.info("No demo images found. Please add images to demo_images/ folder.")
        
        if uploaded_image:
            st.image(uploaded_image, caption="Input Image", use_container_width=True)
    
    # Process button
    if uploaded_image:
        process_btn = st.button("🔍 Detect Parking Spaces", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("Processing..."):
                # Convert PIL image to numpy array
                image_np = np.array(uploaded_image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Initialize detector based on selected backend
                try:
                    if use_roboflow:
                        from utils.roboflow_detector import RoboflowDetector
                        detector = RoboflowDetector(
                            api_key=roboflow_api_key,
                            confidence_threshold=confidence_threshold
                        )
                    else:
                        from utils.detector import ParkingDetector
                        detector = ParkingDetector(
                            model_path=model_path,
                            confidence_threshold=confidence_threshold,
                            device=device
                        )
                except Exception as e:
                    st.error(f"Error loading detector: {e}")
                    st.stop()
                
                # Initialize slot matcher
                if config_file:
                    # Use uploaded config
                    config_dict = yaml.safe_load(config_file)
                    matcher = SlotMatcher(config_dict=config_dict, iou_threshold=iou_threshold)
                    st.info(f"📁 Using uploaded config: {len(config_dict.get('slots', []))} slots")

                elif use_auto_detect:
                    # AUTO-DETECT MODE: Analyze image and create slots automatically
                    st.info("🤖 Auto-detecting parking slots from your image...")

                    auto_detector = AutoSlotDetector()
                    detected_slots, grid_params = auto_detector.detect_slots(image_bgr, force_grid=False)

                    st.success(f"✅ Auto-detected {len(detected_slots)} parking slots!")
                    with st.expander("ℹ️ Auto-Detection Details"):
                        st.json({
                            "Total Slots": len(detected_slots),
                            "Grid Size": f"{grid_params.get('rows', '?')} rows × {grid_params.get('cols', '?')} columns",
                            "Slot Dimensions": f"{grid_params.get('slot_width', '?')}px × {grid_params.get('slot_height', '?')}px",
                            "Starting Position": f"({grid_params.get('start_x', '?')}, {grid_params.get('start_y', '?')})",
                            "Spacing": f"H: {grid_params.get('h_spacing', '?')}px, V: {grid_params.get('v_spacing', '?')}px"
                        })

                    # Convert to slot matcher format
                    matcher = SlotMatcher(iou_threshold=iou_threshold)
                    slot_dicts = auto_detector.slots_to_dict_list(detected_slots)
                    matcher.set_slots_from_list(slot_dicts)

                elif use_manual_grid:
                    # Manual grid mode
                    matcher = SlotMatcher(iou_threshold=iou_threshold)
                    manual_slots = create_grid_slots(
                        start_x=int(image_np.shape[1] * 0.05),
                        start_y=int(image_np.shape[0] * 0.1),
                        slot_width=int(image_np.shape[1] * 0.12),
                        slot_height=int(image_np.shape[0] * 0.18),
                        rows=manual_rows,
                        cols=manual_cols,
                        h_spacing=int(image_np.shape[1] * 0.02),
                        v_spacing=int(image_np.shape[0] * 0.03)
                    )
                    matcher.set_slots_from_list(manual_slots)
                    st.info(f"🎛️ Using manual grid: {manual_rows}×{manual_cols} = {len(manual_slots)} slots")

                else:
                    st.warning("No slot configuration provided.")
                    st.stop()
                
                # Run detection
                detections = detector.detect(image_bgr)
                
                # Match to slots
                result = matcher.match(detections)
                
                # Visualize
                visualizer = ParkingVisualizer()
                
                # Create annotated image
                annotated = visualizer.draw_full_overlay(
                    image_bgr, detections, result,
                    show_detections=show_detections,
                    show_slot_ids=show_slot_ids
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Display results
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(annotated_rgb, caption="Annotated Image", use_container_width=True)
                
                # Schematic map
                if show_schematic:
                    st.subheader("🗺️ Parking Map")
                    schematic = visualizer.create_schematic_map(
                        result.slots,
                        width=800,
                        height=500,
                        title=f"Parking Status - {datetime.now().strftime('%H:%M:%S')}"
                    )
                    schematic_rgb = cv2.cvtColor(schematic, cv2.COLOR_BGR2RGB)
                    st.image(schematic_rgb, caption="Schematic View", use_container_width=True)
                
                # Statistics
                st.subheader("📊 Statistics")
                
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.metric(
                        label="Total Slots",
                        value=result.total_slots
                    )
                
                with stat_cols[1]:
                    st.metric(
                        label="Empty Slots",
                        value=result.empty_slots,
                        delta=None
                    )
                
                with stat_cols[2]:
                    st.metric(
                        label="Occupied Slots",
                        value=result.occupied_slots
                    )
                
                with stat_cols[3]:
                    st.metric(
                        label="Occupancy Rate",
                        value=f"{result.occupancy_rate:.0%}"
                    )
                
                # Progress bar for occupancy
                st.progress(result.occupancy_rate, text=f"Lot is {result.occupancy_rate:.0%} full")
                
                # Slot details
                with st.expander("📋 Slot Details"):
                    # Create a table of slot statuses
                    slot_data = []
                    for slot_id, slot in result.slots.items():
                        slot_data.append({
                            "Slot ID": slot_id,
                            "Status": "🔴 Occupied" if slot.status == SlotStatus.OCCUPIED else "🟢 Empty",
                            "Type": slot.slot_type.capitalize(),
                            "Confidence": f"{slot.occupancy_confidence:.0%}" if slot.status == SlotStatus.OCCUPIED else "-"
                        })
                    
                    st.dataframe(
                        slot_data,
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Detection details
                with st.expander("🚗 Detection Details"):
                    if detections:
                        det_data = []
                        for i, det in enumerate(detections, 1):
                            det_data.append({
                                "#": i,
                                "Class": det.class_name,
                                "Confidence": f"{det.confidence:.0%}",
                                "Location": f"({det.center[0]}, {det.center[1]})",
                                "Size": f"{det.bbox[2]-det.bbox[0]}x{det.bbox[3]-det.bbox[1]}"
                            })
                        st.dataframe(det_data, use_container_width=True, hide_index=True)
                    else:
                        st.info("No vehicles detected in the image.")
                
                # JSON export
                with st.expander("💾 Export Results"):
                    json_result = {
                        "timestamp": datetime.now().isoformat(),
                        "detections_count": len(detections),
                        "occupancy": result.to_dict()
                    }
                    st.json(json_result)
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(json_result, indent=2),
                        file_name=f"parking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    else:
        with col2:
            st.subheader("🎯 Detection Results")
            st.info("👆 Upload an image or select a demo image to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Smart Parking Detection System MVP | Built with YOLOv8 + Streamlit</p>
        <p>For best results, use top-down/overhead images of parking lots</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
