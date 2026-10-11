"""database.py - all MySQL logic (connect, create table, insert, read)."""
import mysql.connector
from mysql.connector import Error


class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.conn = None

    # ---------- setup ----------
    def initialize(self):
        """Create the database + table if missing, then connect to it."""
        server_cfg = {k: v for k, v in self.config.items() if k != "database"}
        tmp = mysql.connector.connect(**server_cfg)
        cur = tmp.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']}")
        tmp.commit()
        cur.close()
        tmp.close()

        self._connect()
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                log_id       INT AUTO_INCREMENT PRIMARY KEY,
                `timestamp`  DATETIME DEFAULT CURRENT_TIMESTAMP,
                object_class VARCHAR(50) NOT NULL,
                confidence   FLOAT NOT NULL,
                bbox_x INT NOT NULL, bbox_y INT NOT NULL,
                bbox_w INT NOT NULL, bbox_h INT NOT NULL
            )
        """)
        self.conn.commit()
        cur.close()

    def _connect(self):
        self.conn = mysql.connector.connect(**self.config)

    def _ensure_connection(self):
        if self.conn is None or not self.conn.is_connected():
            self._connect()

    # ---------- write ----------
    def insert_detections(self, detections):
        """Insert a list of detection dicts. Returns number of rows saved."""
        if not detections:
            return 0
        self._ensure_connection()
        sql = """INSERT INTO detection_logs
                 (object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        rows = [(d["object_class"], d["confidence"], d["bbox_x"], d["bbox_y"],
                 d["bbox_w"], d["bbox_h"]) for d in detections]
        try:
            cur = self.conn.cursor()
            cur.executemany(sql, rows)
            self.conn.commit()
            cur.close()
            return len(rows)
        except Error as e:
            print(f"[DB] insert failed: {e}")
            return 0

    # ---------- read ----------
    def get_recent_logs(self, limit=10):
        self._ensure_connection()
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM detection_logs ORDER BY log_id DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        return rows

    def count_logs(self):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM detection_logs")
        n = cur.fetchone()[0]
        cur.close()
        return n

    def clear_logs(self):
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute("TRUNCATE TABLE detection_logs")
        self.conn.commit()
        cur.close()
