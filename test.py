import cv2
import os
import mysql.connector
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ---------------------------------------- #
#           MYSQL DATABASE SETUP
# ---------------------------------------- #
def get_mysql_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "mysql"),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", "root"),
        database=os.environ.get("MYSQL_DB", "marathon_bibs")
    )

def init_mysql():
    conn = get_mysql_connection()
    cur = conn.cursor()

    # Create schema
    cur.execute("CREATE DATABASE IF NOT EXISTS marathon_bibs")
    cur.execute("USE marathon_bibs")

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bibs_detected (
            id INT AUTO_INCREMENT PRIMARY KEY,
            bib_number VARCHAR(255),
            img_path VARCHAR(255)
        )
    """)

    conn.commit()
    conn.close()

def insert_bib_mysql(bib_number, img_path):
    conn = get_mysql_connection()
    cur = conn.cursor()
    cur.execute("USE marathon_bibs")

    sql = """
        INSERT INTO bibs_detected (bib_number, img_path)
        VALUES (%s, %s)
    """
    cur.execute(sql, (bib_number, img_path))

    conn.commit()
    conn.close()

# Initialize DB on startup
init_mysql()

# ---------------------------------------- #
#           MODEL LOADING
# ---------------------------------------- #
person_model = YOLO("models/yolov8n.pt")
bib_model = YOLO("models/git_bib.pt")

ocr = PaddleOCR(
    use_angle_cls=True,
    show_log=False,
    det_model_dir="PP-OCRv5_server_det",
    rec_model_dir="PP-OCRv5_mobile_rec",
    lang="en"
)



# ---------------------------------------- #
#              CONSTANTS
# ---------------------------------------- #
PERSON_CONF = 0.60
BIB_CONF = 0.15
BIB_PAD_W = 40
BIB_PAD_H = 0

# ---------------------------------------- #
#              UTILITIES
# ---------------------------------------- #
def to_gray_3c(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

# ---------------------------------------- #
#         DETECT FOLDER WRAPPER
# ---------------------------------------- #
def detect_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in os.listdir(input_folder)
              if f.lower().endswith(valid_ext)]

    print(f"Processing {len(images)} images...\n")

    for idx, f in enumerate(images):
        inp = os.path.join(input_folder, f)
        out = os.path.join(output_folder, f)

        print(f"[{idx+1}/{len(images)}] Processing: {f}")
        process_single_image(inp, out)

# ---------------------------------------- #
#            PROCESS IMAGE
# ---------------------------------------- #
def process_single_image(image_path, output_path):

    img_original = cv2.imread(image_path)
    img_gray = to_gray_3c(img_original)

    # PERSON DETECTION
    person_results = person_model(img_gray)[0]
    person_boxes = []

    for box in person_results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf >= PERSON_CONF:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    if not person_boxes:
        cv2.imwrite(output_path, img_original)
        print("  No persons found.")
        return

    # BIB DETECTION
    bib_counter = 0

    for (px1, py1, px2, py2) in person_boxes:

        cv2.rectangle(img_original, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(img_original, "person", (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        roi = img_gray[py1:py2, px1:px2]
        roi_h, roi_w = roi.shape[:2]

        bib_results = bib_model(roi)[0]

        for box in bib_results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf >= BIB_CONF:

                bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                pbx1 = max(0, bx1 - BIB_PAD_W)
                pbx2 = min(roi_w, bx2 + BIB_PAD_W)
                pby1 = max(0, by1 - BIB_PAD_H)
                pby2 = min(roi_h, by2 + BIB_PAD_H)

                gx1, gy1 = px1 + pbx1, py1 + pby1
                gx2, gy2 = px1 + pbx2, py1 + pby2

                crop = img_original[gy1:gy2, gx1:gx2]

                # OCR with angles
                candidates = [
                    crop,
                    rotate_image(crop, 15),
                    rotate_image(crop, -15)
                ]

                best_text = "N/A"
                best_conf = 0

                for test in candidates:
                    res = ocr.ocr(test, cls=True)
                    if res and res[0] and res[0][0]:
                        t = res[0][0][1][0]
                        c = res[0][0][1][1]
                        if c > best_conf:
                            best_conf = c
                            best_text = t

                ocr_text = best_text.strip()

                print(f"  Bib {bib_counter}: {ocr_text} ({best_conf:.2f})")

                # INSERT INTO MYSQL
                if ocr_text != "N/A" and ocr_text != "":
                    insert_bib_mysql(ocr_text, output_path)

                # Annotate
                cv2.rectangle(img_original, (gx1, gy1), (gx2, gy2),
                              (0, 255, 0), 2)
                cv2.putText(img_original, f"{ocr_text} ({best_conf:.2f})",
                            (gx1, gy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

                bib_counter += 1

    cv2.imwrite(output_path, img_original)
    print(f"  Saved annotated → {output_path}\n")

# ---------------------------------------- #
#                RUN
# ---------------------------------------- #
detect_folder(
    input_folder=os.environ.get("INPUT_FOLDER", "input"),
    output_folder=os.environ.get("OUTPUT_FOLDER", "output")
)