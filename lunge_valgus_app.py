"""
Biomechanical Lunge Analysis: Knee Valgus Tracking
====================================================
A single-file Streamlit application that uses OpenCV + MediaPipe Pose to
analyze uploaded lunge videos for dynamic knee valgus (medial knee collapse).

Run locally:
    pip install streamlit opencv-python-headless mediapipe numpy
    streamlit run lunge_valgus_app.py

Deploy on Streamlit Community Cloud:
    Push this file + a requirements.txt containing:
        streamlit
        opencv-python-headless
        mediapipe
        numpy
"""

import tempfile
import os

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# --------------------------------------------------------------------------
# Page / global config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Biomechanical Lunge Analysis",
    layout="wide",
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Pose landmark indices used for the analysis
LANDMARKS = {
    "Left": {"hip": 23, "knee": 25, "ankle": 27},
    "Right": {"hip": 24, "knee": 26, "ankle": 28},
}

COLOR_GREEN = (0, 200, 0)   # BGR - aligned / safe
COLOR_RED = (0, 0, 255)     # BGR - valgus warning
COLOR_WHITE = (255, 255, 255)


# --------------------------------------------------------------------------
# Biomechanical helper functions
# --------------------------------------------------------------------------
def get_point(landmarks, index, frame_w, frame_h):
    """Return (x_px, y_px, visibility) for a given MediaPipe landmark index."""
    lm = landmarks[index]
    return np.array([lm.x * frame_w, lm.y * frame_h]), lm.visibility


def compute_medial_offset(hip_xy, knee_xy, ankle_xy, leg_side):
    """
    Compute the horizontal (x-axis) deviation of the knee from the
    hip-ankle reference line, signed so that a POSITIVE value always
    means the knee has drifted toward the body's medial midline
    (i.e. classic knee valgus / medial collapse).

    Method:
      1. Build the hip->ankle line.
      2. Interpolate the line's x-position at the knee's y-height
         (this is the "expected" knee x if the leg were perfectly straight).
      3. offset = actual_knee_x - expected_x.
      4. Sign-correct based on which leg is active, since medial direction
         is mirrored between the left and right leg on screen.

    Assumption: the camera faces the subject directly (front-on, not a
    mirrored/selfie view). With that orientation, the subject's LEFT leg
    appears on the RIGHT side of the frame (medial = decreasing x), and the
    subject's RIGHT leg appears on the LEFT side of the frame
    (medial = increasing x).
    """
    hip_x, hip_y = hip_xy
    knee_x, knee_y = knee_xy
    ankle_x, ankle_y = ankle_xy

    if abs(ankle_y - hip_y) > 1e-6:
        t = (knee_y - hip_y) / (ankle_y - hip_y)
        expected_x = hip_x + t * (ankle_x - hip_x)
    else:
        # Degenerate case: hip and ankle at same height, fall back to hip x
        expected_x = hip_x

    raw_offset = knee_x - expected_x

    if leg_side == "Left":
        medial_offset = -raw_offset
    else:
        medial_offset = raw_offset

    return medial_offset, expected_x


# --------------------------------------------------------------------------
# Sidebar - user controls
# --------------------------------------------------------------------------
st.sidebar.header("Analysis Settings")

valgus_threshold = st.sidebar.slider(
    "Knee Valgus Warning Threshold (px)",
    min_value=10,
    max_value=100,
    value=30,
    step=1,
    help="If the knee drifts medially past the hip-ankle line by more than "
         "this many pixels, the frame is flagged as valgus.",
)

visibility_confidence = st.sidebar.slider(
    "Landmark Visibility Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum MediaPipe landmark visibility/detection confidence "
         "required for a frame to be analyzed.",
)

lead_leg = st.sidebar.selectbox(
    "Active Lead Leg",
    options=["Left", "Right"],
    index=0,
    help="Which leg is performing the lunge and should be tracked.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Assumes a front-facing, non-mirrored camera view of the subject."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Lunge Video",
    type=["mp4", "mov", "avi"],
)

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title("Biomechanical Lunge Analysis: Knee Valgus Tracking")
st.write(
    "Upload a video of a dynamic lunge to track medial knee collapse "
    "(knee valgus) relative to the hip-ankle reference line, in real time."
)

if uploaded_file is not None:
    # Persist the uploaded video to a temp file so OpenCV can read it
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tmp_input.write(uploaded_file.read())
    tmp_input.close()

    start_analysis = st.button("Run Analysis", type="primary")

    if start_analysis:
        cap = cv2.VideoCapture(tmp_input.name)

        if not cap.isOpened():
            st.error("Could not open the uploaded video file. Please try a different file.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            processed_frames = 0
            valgus_frames = 0
            frame_index = 0

            hip_idx = LANDMARKS[lead_leg]["hip"]
            knee_idx = LANDMARKS[lead_leg]["knee"]
            ankle_idx = LANDMARKS[lead_leg]["ankle"]

            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=visibility_confidence,
                min_tracking_confidence=visibility_confidence,
            ) as pose:

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    frame_index += 1
                    frame_h, frame_w = frame.shape[:2]

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    results = pose.process(rgb_frame)
                    rgb_frame.flags.writeable = True

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark

                        hip_xy, hip_vis = get_point(landmarks, hip_idx, frame_w, frame_h)
                        knee_xy, knee_vis = get_point(landmarks, knee_idx, frame_w, frame_h)
                        ankle_xy, ankle_vis = get_point(landmarks, ankle_idx, frame_w, frame_h)

                        min_visibility = min(hip_vis, knee_vis, ankle_vis)

                        if min_visibility >= visibility_confidence:
                            processed_frames += 1

                            medial_offset, expected_x = compute_medial_offset(
                                hip_xy, knee_xy, ankle_xy, lead_leg
                            )
                            is_valgus = medial_offset > valgus_threshold

                            if is_valgus:
                                valgus_frames += 1
                                line_color = COLOR_RED
                                knee_color = COLOR_RED
                                banner_text = "WARNING: KNEE VALGUS DETECTED"
                                banner_color = COLOR_RED
                            else:
                                line_color = COLOR_GREEN
                                knee_color = COLOR_GREEN
                                banner_text = "ALIGNMENT SAFE"
                                banner_color = COLOR_GREEN

                            hip_pt = tuple(hip_xy.astype(int))
                            knee_pt = tuple(knee_xy.astype(int))
                            ankle_pt = tuple(ankle_xy.astype(int))

                            # Hip -> Ankle reference line
                            cv2.line(frame, hip_pt, ankle_pt, line_color, 3)
                            # Hip / Ankle anchor points
                            cv2.circle(frame, hip_pt, 6, COLOR_WHITE, -1)
                            cv2.circle(frame, ankle_pt, 6, COLOR_WHITE, -1)
                            # Knee landmark (color-coded)
                            cv2.circle(frame, knee_pt, 9, knee_color, -1)

                            # Status banner
                            cv2.rectangle(frame, (0, 0), (frame_w, 50), banner_color, -1)
                            cv2.putText(
                                frame, banner_text, (15, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2, cv2.LINE_AA,
                            )
                            cv2.putText(
                                frame,
                                f"Medial Offset: {medial_offset:.1f}px (Threshold: {valgus_threshold}px)",
                                (15, frame_h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA,
                            )
                        else:
                            cv2.putText(
                                frame, "LOW VISIBILITY: SKIPPING FRAME", (15, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA,
                            )
                    else:
                        cv2.putText(
                            frame, "NO POSE DETECTED", (15, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA,
                        )

                    # Display frame (convert back to RGB for Streamlit)
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                    if total_frame_count > 0:
                        progress_bar.progress(min(frame_index / total_frame_count, 1.0))
                    status_text.text(f"Processing frame {frame_index}/{total_frame_count or '?'}")

            cap.release()
            os.unlink(tmp_input.name)
            progress_bar.empty()
            status_text.empty()

            # ------------------------------------------------------------
            # Summary metrics
            # ------------------------------------------------------------
            st.markdown("---")
            st.subheader("Session Summary")

            valgus_pct = (valgus_frames / processed_frames * 100) if processed_frames > 0 else 0.0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Processed Frames", f"{processed_frames}")
            col2.metric("Frames in Valgus Collapse", f"{valgus_frames}")
            col3.metric("Percentage of Frames in Valgus Collapse", f"{valgus_pct:.1f}%")

            if processed_frames == 0:
                st.warning(
                    "No frames met the visibility confidence threshold. Try lowering "
                    "the 'Landmark Visibility Confidence' slider or use a clearer video."
                )
            elif valgus_pct > 25:
                st.error(
                    "Significant knee valgus detected across a large portion of this rep. "
                    "Consider reviewing lunge mechanics and hip/glute activation."
                )
            elif valgus_pct > 0:
                st.warning(
                    "Some frames showed medial knee collapse. Monitor form under fatigue "
                    "or with heavier load."
                )
            else:
                st.success("No knee valgus detected — alignment remained safe throughout.")
else:
    st.info("Upload a lunge video (.mp4, .mov, .avi) from the sidebar to begin analysis.")
