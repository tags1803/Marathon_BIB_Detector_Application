import cv2
import os
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR


# ----------------------------- #
#       MODEL LOADING
# ----------------------------- #

person_model = YOLO("yolov8n.pt")  
bib_model = YOLO("models/git_bib.pt")  

ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

PERSON_CONF = 0.60
BIB_CONF = 0.20

PAD_W = 40
PAD_H = 10

BIB_SAVE_ROOT = os.environ.get("BIB_SAVE_ROOT", "output/detected_bibs")
os.makedirs(BIB_SAVE_ROOT, exist_ok=True)


# ----------------------------- #
#  VALID BIB NUMBER CHECK LOGIC
# ----------------------------- #

def validate_bib_number(ocr_text):
    """
    Valid bib formats:
      1) 4+ digit numbers (e.g., 1023)
      2) 3+ digits + one letter at end (e.g., 100A, 1000C)
    """

    if ocr_text is None:
        return None

    text = ocr_text.strip().upper()

    # CASE 1: Pure digits, length >= 4
    if text.isdigit() and len(text) >= 4:
        return text

    # CASE 2: digits + final alphabet (min 3 digits)
    if len(text) >= 4 and text[:-1].isdigit() and text[-1].isalpha():
        if len(text[:-1]) >= 3:   # digit part is >= 3
            return text

    return None


# ----------------------------- #
#      UTILITY FUNCTIONS
# ----------------------------- #

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def to_gray_3c(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# ----------------------------- #
#       MAIN VIDEO PIPELINE
# ----------------------------- #

def run_video(input_video_path, output_video_path=None):

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(3))
        h = int(cap.get(4))
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    frame_index = 0
    print("Processing video... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        H, W = frame.shape[:2]

        gray_frame = to_gray_3c(frame)

        # ---------------------------- #
        #  1. PERSON DETECTION
        # ---------------------------- #
        person_results = person_model(gray_frame)[0]
        person_boxes = []

        for box in person_results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= PERSON_CONF:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

        # Track which bibs appear in this frame
        bibs_in_this_frame = set()

        # ---------------------------- #
        #  2. BIB DETECTION INSIDE ROI
        # ---------------------------- #
        for (px1, py1, px2, py2) in person_boxes:

            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "person", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            person_roi = gray_frame[py1:py2, px1:px2]
            roi_h, roi_w = person_roi.shape[:2]

            bib_results = bib_model(person_roi)[0]

            for box in bib_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != 0 or conf < BIB_CONF:
                    continue

                bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                # Padding
                pbx1 = max(0, bx1 - PAD_W)
                pbx2 = min(roi_w, bx2 + PAD_W)
                pby1 = max(0, by1 - PAD_H)
                pby2 = min(roi_h, by2 + PAD_H)

                # Convert to global image coords
                gx1 = px1 + pbx1
                gy1 = py1 + pby1
                gx2 = px1 + pbx2
                gy2 = py1 + pby2

                crop = frame[gy1:gy2, gx1:gx2]

                # OCR with tilt correction
                candidates = [
                    crop,
                    rotate_image(crop, +15),
                    rotate_image(crop, -15)
                ]

                best_text = "N/A"
                best_conf = 0

                for test_crop in candidates:
                    ocr_res = ocr.ocr(test_crop, cls=True)
                    if ocr_res and ocr_res[0] and ocr_res[0][0]:
                        line = ocr_res[0][0]
                        txt = line[1][0]
                        t_conf = line[1][1]
                        if t_conf > best_conf:
                            best_conf = t_conf
                            best_text = txt

                # Validate bib format
                valid_bib = validate_bib_number(best_text)

                # Draw result
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{best_text} ({best_conf:.2f})",
                            (gx1, gy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

                # Save if valid bib detected
                if valid_bib:
                    bibs_in_this_frame.add(valid_bib)

        # ---------------------------- #
        #  3. SAVE FRAME PER BIB
        # ---------------------------- #
        for bib in bibs_in_this_frame:
            save_dir = os.path.join(BIB_SAVE_ROOT, f"bib_{bib}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(save_path, frame)

        # ---------------------------- #
        #  DISPLAY & SAVE
        # ---------------------------- #
        cv2.imshow("Bib Detection", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("Video processing completed.")


# ----------------------------- #
#       RUN PIPELINE
# ----------------------------- #

run_video(
    input_video_path=os.environ.get("INPUT_VIDEO", "input/video.mp4"),
    output_video_path=os.environ.get("OUTPUT_VIDEO", "output/annotated_video.mp4")
)