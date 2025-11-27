"""
Smart Parking Detection - Interactive Slot Annotator
----------------------------------------------------
Web-based tool to create custom parking slot configurations.

Run with:
    streamlit run app/slot_annotator.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import base64
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yaml
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Parking Slot Annotator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_slot_from_rect(x, y, width, height, slot_id):
    """Create a slot dictionary from rectangle coordinates."""
    return {
        'id': slot_id,
        'polygon': [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ],
        'type': 'regular'
    }


def generate_grid_slots(image_width, image_height, params):
    """Generate parking slots in a grid pattern."""
    slots = []
    slot_counter = 0

    for row in range(params['rows']):
        row_letter = chr(ord('A') + row)
        y = params['start_y'] + row * (params['slot_height'] + params['v_spacing'])

        for col in range(params['cols']):
            x = params['start_x'] + col * (params['slot_width'] + params['h_spacing'])

            # Check if slot is within image bounds
            if (x + params['slot_width'] <= image_width and
                y + params['slot_height'] <= image_height):

                slot = {
                    'id': f"{row_letter}{col + 1}",
                    'polygon': [
                        [int(x), int(y)],
                        [int(x + params['slot_width']), int(y)],
                        [int(x + params['slot_width']), int(y + params['slot_height'])],
                        [int(x), int(y + params['slot_height'])]
                    ],
                    'type': 'regular'
                }
                slots.append(slot)
                slot_counter += 1

    return slots


def draw_slots_on_image(image, slots, show_ids=True):
    """Draw parking slots on image."""
    overlay = image.copy()

    for slot in slots:
        pts = np.array(slot['polygon'], np.int32).reshape((-1, 1, 2))

        # Draw filled polygon
        cv2.fillPoly(overlay, [pts], (0, 255, 0), lineType=cv2.LINE_AA)

        # Draw border
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        if show_ids:
            # Calculate center
            center = np.mean(slot['polygon'], axis=0).astype(int)

            # Draw slot ID with background
            text = slot['id']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Background rectangle
            cv2.rectangle(overlay,
                         (center[0] - text_width//2 - 5, center[1] - text_height//2 - 5),
                         (center[0] + text_width//2 + 5, center[1] + text_height//2 + 5),
                         (0, 0, 0), -1)

            # Text
            cv2.putText(overlay, text,
                       (center[0] - text_width//2, center[1] + text_height//2),
                       font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Blend with original image
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    return result


def main():
    st.title("🎯 Parking Slot Annotator")
    st.markdown("""
    Create custom parking slot configurations that match your parking lot layout.
    Upload an image and define slots using the grid generator or manual annotation.
    """)

    # Initialize session state
    if 'slots' not in st.session_state:
        st.session_state.slots = []
    if 'image_uploaded' not in st.session_state:
        st.session_state.image_uploaded = False

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Image upload
    st.sidebar.subheader("1️⃣ Upload Image")

    # Demo images section
    demo_dir = Path(__file__).parent.parent / "demo_images"
    demo_images = list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png")) if demo_dir.exists() else []

    input_method = st.sidebar.radio(
        "Choose input:",
        options=["Upload Image", "Use Demo Image"],
        horizontal=False
    )

    uploaded_image = None
    image_name = None

    if input_method == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Select parking lot image",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            image_name = uploaded_file.name
            st.session_state.image_uploaded = True

    elif input_method == "Use Demo Image":
        if demo_images:
            selected_demo = st.sidebar.selectbox(
                "Select demo image:",
                options=[p.name for p in demo_images]
            )
            demo_path = demo_dir / selected_demo
            uploaded_image = Image.open(demo_path)
            image_name = selected_demo
            st.session_state.image_uploaded = True
        else:
            st.sidebar.info("No demo images found.")

    if uploaded_image:
        image_np = np.array(uploaded_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        img_height, img_width = image_bgr.shape[:2]

        # Sidebar - Slot Generation
        st.sidebar.subheader("2️⃣ Generate Slots")

        generation_mode = st.sidebar.radio(
            "Generation mode:",
            options=["Auto Grid", "Custom Grid", "Manual Draw"],
            help="Auto Grid: Quick preset layouts | Custom Grid: Adjust parameters | Manual Draw: Click to draw"
        )

        if generation_mode == "Auto Grid":
            st.sidebar.markdown("**Quick Presets:**")

            preset = st.sidebar.selectbox(
                "Select layout preset:",
                options=[
                    "3x5 - Small Lot (15 spots)",
                    "4x6 - Medium Lot (24 spots)",
                    "5x8 - Large Lot (40 spots)",
                    "6x10 - XL Lot (60 spots)",
                    "Custom"
                ]
            )

            # Parse preset
            if preset == "3x5 - Small Lot (15 spots)":
                rows, cols = 3, 5
            elif preset == "4x6 - Medium Lot (24 spots)":
                rows, cols = 4, 6
            elif preset == "5x8 - Large Lot (40 spots)":
                rows, cols = 5, 8
            elif preset == "6x10 - XL Lot (60 spots)":
                rows, cols = 6, 10
            else:
                rows = st.sidebar.number_input("Rows", 1, 20, 4)
                cols = st.sidebar.number_input("Columns", 1, 20, 6)

            # Auto-calculate slot dimensions based on image size
            available_width = int(img_width * 0.85)
            available_height = int(img_height * 0.80)

            slot_width = int(available_width / cols) - 15
            slot_height = int(available_height / rows) - 25

            params = {
                'start_x': int(img_width * 0.075),
                'start_y': int(img_height * 0.10),
                'slot_width': slot_width,
                'slot_height': slot_height,
                'rows': rows,
                'cols': cols,
                'h_spacing': 15,
                'v_spacing': 25
            }

        elif generation_mode == "Custom Grid":
            st.sidebar.markdown("**Adjust Grid Parameters:**")

            col1, col2 = st.sidebar.columns(2)
            with col1:
                rows = st.number_input("Rows", 1, 20, 4, key="custom_rows")
                cols = st.number_input("Cols", 1, 20, 6, key="custom_cols")

            with col2:
                h_spacing = st.number_input("H-Space", 0, 100, 15, key="h_space")
                v_spacing = st.number_input("V-Space", 0, 100, 25, key="v_space")

            st.sidebar.markdown("**Position & Size:**")

            start_x = st.sidebar.slider("Start X", 0, img_width, int(img_width * 0.05), key="start_x")
            start_y = st.sidebar.slider("Start Y", 0, img_height, int(img_height * 0.10), key="start_y")

            slot_width = st.sidebar.slider("Slot Width", 20, 300,
                                          int(img_width * 0.08), key="slot_w")
            slot_height = st.sidebar.slider("Slot Height", 20, 300,
                                           int(img_height * 0.15), key="slot_h")

            params = {
                'start_x': start_x,
                'start_y': start_y,
                'slot_width': slot_width,
                'slot_height': slot_height,
                'rows': rows,
                'cols': cols,
                'h_spacing': h_spacing,
                'v_spacing': v_spacing
            }

        else:  # Manual Draw
            st.sidebar.info("👆 Click and drag on the image to draw parking slots")
            params = None

        # Generate slots button (for grid modes)
        if generation_mode in ["Auto Grid", "Custom Grid"]:
            if st.sidebar.button("🎯 Generate Slots", type="primary", use_container_width=True):
                st.session_state.slots = generate_grid_slots(img_width, img_height, params)
                st.success(f"✅ Generated {len(st.session_state.slots)} slots!")
                st.rerun()

        # Actions
        st.sidebar.subheader("3️⃣ Actions")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.slots = []
                st.rerun()

        with col2:
            show_ids = st.checkbox("Show IDs", value=True)

        # Save configuration
        if st.session_state.slots:
            lot_name = st.sidebar.text_input(
                "Lot Name",
                value=f"Parking Lot - {image_name}",
                key="lot_name"
            )

            if st.sidebar.button("💾 Save Configuration", type="primary", use_container_width=True):
                config = {
                    'lot_name': lot_name,
                    'description': f'Created from {image_name} on {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    'image_width': img_width,
                    'image_height': img_height,
                    'detection': {
                        'confidence_threshold': 0.25,
                        'iou_threshold': 0.3,
                        'model': 'yolov8m.pt'
                    },
                    'slots': st.session_state.slots,
                    'visualization': {
                        'empty_color': [0, 255, 0],
                        'occupied_color': [0, 0, 255],
                        'unknown_color': [128, 128, 128],
                        'slot_alpha': 0.4,
                        'border_thickness': 2,
                        'font_scale': 0.5
                    }
                }

                # Save to configs directory
                config_path = Path(__file__).parent.parent / "configs" / f"{lot_name.lower().replace(' ', '_')}.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=None, sort_keys=False)

                st.sidebar.success(f"✅ Saved to: {config_path.name}")

                # Also offer download
                yaml_str = yaml.dump(config, default_flow_style=None, sort_keys=False)
                st.sidebar.download_button(
                    label="📥 Download YAML",
                    data=yaml_str,
                    file_name=f"{lot_name.lower().replace(' ', '_')}.yaml",
                    mime="text/yaml"
                )

        # Main content area
        st.subheader("📸 Parking Lot Image with Slots")

        # Display info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Image Size", f"{img_width}x{img_height}")
        with col2:
            st.metric("Total Slots", len(st.session_state.slots))
        with col3:
            if st.session_state.slots:
                st.metric("Status", "✅ Ready to Save", delta="Configured")
            else:
                st.metric("Status", "⏳ Awaiting Setup", delta="Not Configured")

        # Draw slots on image
        if st.session_state.slots:
            annotated_image = draw_slots_on_image(image_bgr.copy(), st.session_state.slots, show_ids)
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=f"Configured Layout ({len(st.session_state.slots)} slots)", use_column_width=True)

            # Show slot details
            with st.expander("📋 Slot Details"):
                slot_data = []
                for slot in st.session_state.slots:
                    bounds = slot['polygon']
                    x_coords = [p[0] for p in bounds]
                    y_coords = [p[1] for p in bounds]

                    slot_data.append({
                        "ID": slot['id'],
                        "Type": slot['type'].capitalize(),
                        "X": f"{min(x_coords)}-{max(x_coords)}",
                        "Y": f"{min(y_coords)}-{max(y_coords)}"
                    })

                st.dataframe(slot_data, use_container_width=True, hide_index=True)
        else:
            st.image(image_np, caption="Original Image (No slots configured)", use_column_width=True)
            st.info("👈 Use the sidebar to generate parking slots")

    else:
        st.info("👈 Upload an image or select a demo image to get started")

        # Show instructions
        st.markdown("""
        ### 📖 How to Use

        1. **Upload Image**: Choose a top-down parking lot image
        2. **Generate Slots**: Use one of three modes:
           - **Auto Grid**: Quick presets for common layouts
           - **Custom Grid**: Fine-tune grid parameters
           - **Manual Draw**: Draw slots by hand (coming soon)
        3. **Adjust**: Preview and adjust slot positions
        4. **Save**: Download YAML configuration file
        5. **Use**: Upload the YAML in the main detection app

        ### 💡 Tips

        - Start with Auto Grid for quick setup
        - Use Custom Grid for precise control
        - Adjust spacing to match lane widths
        - Save multiple configurations for different lots
        - Test with the main app to verify alignment
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Parking Slot Annotator | Create custom parking lot configurations</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
