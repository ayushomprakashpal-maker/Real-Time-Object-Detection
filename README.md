# Real-Time Object Detection Platform

A real-time object detection dashboard built with **Python, OpenCV, YOLOv8, MySQL and Streamlit**.
It reads a webcam or video file, draws bounding boxes live, and logs detection events to MySQL.

## Features
- Live webcam / video-file detection with YOLOv8n
- Filter which objects to detect (e.g. only `person`, `cell phone`)
- Adjustable confidence threshold
- Throttled event logging to MySQL (`vision_platform.detection_logs`)
- Dashboard with live feed, FPS / detection metrics and latest DB logs

## Project structure
```
main.py              # entry point, real-time loop
detector.py          # YOLO detection + drawing
database.py          # MySQL connection and queries
ui.py                # Streamlit interface components
config.py            # loads settings from .env
database_schema.sql  # DB + table definition
requirements.txt
.env.example         # template for your credentials
```

## Setup
1. Install Python 3.9+ and MySQL (XAMPP / WAMP / MySQL Server).
2. Start MySQL.
3. Clone and install:
   ```bash
   git clone https://github.com/<your-username>/Real-Time-Object-Detection.git
   cd Real-Time-Object-Detection
   python -m venv venv
   venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your MySQL password.
5. (Optional) Create the DB manually: `mysql -u root -p < database_schema.sql`
   (the app also creates it automatically on first run).
6. Run:
   ```bash
   streamlit run main.py
   ```

## Database schema
| Column | Type | Description |
|---|---|---|
| log_id | INT PK AUTO_INCREMENT | Unique ID |
| timestamp | DATETIME | When detected |
| object_class | VARCHAR(50) | e.g. person |
| confidence | FLOAT | 0–1 score |
| bbox_x, bbox_y, bbox_w, bbox_h | INT | Bounding box |

## Demo
- Video: <add your link>
- Medium article: <add your link>

## Author
<Your name>
