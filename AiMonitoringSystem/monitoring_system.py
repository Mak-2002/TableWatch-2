from datetime import datetime, timedelta

import cv2
import cvzone
from ultralytics import YOLO

from config import VIDEO_FILES, CURRENT_VIDEO_INDEX, WAITER_IMAGES_PATH, PLAYBACK_SPEED, CLASS_COLORS, \
    PROBLEM_CLIPS_PATH
from database import connect_to_database, log_detected_issue
from face_recognition_module import load_waiter_images, encode_waiter_faces, detect_and_log_faces
from object_detection_module import process_video_frame
from utils import are_tables_similar


class RestaurantMonitoringSystem:
    def __init__(self, playback_speed=PLAYBACK_SPEED, display_detection_logs=True):
        self.database_connection = connect_to_database()
        self.database_cursor = self.database_connection.cursor(buffered=True)

        self.video_capture = cv2.VideoCapture(VIDEO_FILES[CURRENT_VIDEO_INDEX])
        self.object_detection_model = YOLO("../YOLOv8/Yolo-Weights/yolov8x.pt")

        self.playback_speed = playback_speed
        self.display_detection_logs = display_detection_logs

        self.waiter_images, self.waiter_names = load_waiter_images(WAITER_IMAGES_PATH)
        self.known_face_encodings = encode_waiter_faces(self.waiter_images)

        self.table_counter = 0
        self.table_identifiers = {}

        self.last_detection_time = {}
        self.last_unreserved_log_time = {}

        self.cooldown_period = timedelta(minutes=5)  # Set cooldown period to 5 minutes

    def assign_table_identifier(self, coords):
        """Assign a unique number to each detected table."""
        for existing_coords, table_number in self.table_identifiers.items():
            if are_tables_similar(existing_coords, coords):
                return table_number

        self.table_counter += 1
        self.table_identifiers[coords] = self.table_counter
        return self.table_counter

    def check_for_expired_reservations(self, table_number, table_coords, img):
        """Check if a reservation has expired and log it if so."""
        now = datetime.now()

        query = """
                SELECT 
                    reservations.id, 
                    reservations.date, 
                    reservations.timeStart, 
                    reservations.timeEnd, 
                    reservations.notification_sent, 
                    tables.number AS table_number 
                FROM reservations 
                INNER JOIN tables 
                    ON reservations.tableID = tables.id 
                WHERE tables.number = %s
            """
        self.database_cursor.execute(query, (table_number,))
        reservations = self.database_cursor.fetchall()

        for reservation in reservations:
            reservation_id = reservation[0]
            reservation_date = datetime.strptime(reservation[1], "%Y-%m-%d").date()
            reservation_start_time = datetime.strptime(str(reservation[2]), "%H:%M:%S").time()
            notification_sent = reservation[4]

            threshold = 15
            expiry_time = datetime.combine(reservation_date, reservation_start_time) + timedelta(minutes=threshold)

            if now >= expiry_time and not notification_sent:
                issue_description = (
                    f"Reservation #{reservation_id} for Table {table_number} at {reservation_start_time} "
                    f"is not occupied for {threshold} minutes.")
                video_clip_path = f"reservation_expired_{table_number}_{reservation_id}.mp4"

                self.handle_problem_logging(
                    problem_name="Reservation Expired",
                    problem_description=issue_description,
                    video_clip_path=video_clip_path,
                    table_number=table_number,
                    reservation_id=reservation_id
                )

                update_query = """
                        UPDATE reservations 
                        SET notification_sent = TRUE 
                        WHERE id = %s
                    """
                self.database_cursor.execute(update_query, (reservation_id,))
                self.database_connection.commit()

    def monitor_table_and_food_status(self, img, tables, people, food_items):
        """Monitor each detected table and handle food delivery issues."""
        for coords in tables:
            x1, y1, x2, y2, table_number = coords

            table_occupied = any(
                px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1
                for (px1, py1, px2, py2) in people
            )

            self.handle_table_occupancy(table_number, coords, img, table_occupied)

            if table_occupied and not any(
                    px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1
                    for (px1, py1, px2, py2) in food_items
            ):
                self.handle_delayed_food_delivery(table_number, coords, img)

            status = f"Occupied (Table {table_number})" if table_occupied else f"Vacant (Table {table_number})"
            cvzone.putTextRect(
                img,
                status,
                (x1, y1 - 35),
                scale=0.6,
                thickness=1,
                colorR=CLASS_COLORS['dining table'],
            )

    def handle_delayed_food_delivery(self, table_number, coords, img):
        """Handle the detection and logging of delayed food delivery issues."""
        current_time = datetime.now()

        query = """
            SELECT reservations.date, reservations.timeStart, reservations.timeEnd, tables.number AS table_number 
            FROM reservations 
            INNER JOIN tables ON reservations.tableID = tables.id 
            WHERE tables.number = %s
            AND %s BETWEEN reservations.timeStart AND reservations.timeEnd
        """
        self.database_cursor.execute(query, (table_number, current_time.time()))
        reservation = self.database_cursor.fetchone()

        if reservation:
            date_start = datetime.strptime(reservation[0], "%Y-%m-%d").date()
            time_start = datetime.strptime(str(reservation[1]), "%H:%M:%S").time()
            reservation_start_time = datetime.combine(date_start, time_start)

            if current_time > (reservation_start_time + timedelta(seconds=15)):  # TODO: Change to minutes in production
                last_detected = self.last_detection_time.get(table_number)

                if last_detected is None or (current_time - last_detected) > timedelta(minutes=10):
                    issue_description = (
                        f"Table {table_number} at {coords} has been occupied for a long time "
                        "without food delivery.")
                    video_clip_path = f"delayed_food_delivery_{table_number}.mp4"

                    self.handle_problem_logging(
                        problem_name="Delayed Food Delivery",
                        problem_description=issue_description,
                        video_clip_path=video_clip_path,
                        table_number=table_number
                    )

                    self.last_detection_time[table_number] = current_time

    def handle_table_occupancy(self, table_number, coords, img, table_occupied):
        now = datetime.now()
        if table_occupied:
            query = """
            SELECT COUNT(*) FROM reservations 
            INNER JOIN tables ON reservations.tableID = tables.id 
            WHERE tables.number = %s AND reservations.date = CURDATE()
            AND CURTIME() BETWEEN reservations.timeStart AND reservations.timeEnd
            """
            self.database_cursor.execute(query, (table_number,))
            reservation_count = self.database_cursor.fetchone()[0]

            if reservation_count == 0:
                last_log_time = self.last_unreserved_log_time.get(table_number)
                if last_log_time is None or (now - last_log_time) > self.cooldown_period:
                    issue_description = f"Table {table_number} is occupied but not reserved."
                    video_clip_path = f"unreserved_occupied_{table_number}.mp4"

                    self.handle_problem_logging(
                        problem_name="Unreserved Table Occupied",
                        problem_description=issue_description,
                        video_clip_path=video_clip_path,
                        table_number=table_number
                    )

                    self.last_unreserved_log_time[table_number] = now
        else:
            self.check_for_expired_reservations(table_number, coords, img)

    def handle_problem_logging(self, problem_name, problem_description, video_clip_path, table_number,
                               reservation_id=None, clip_duration=10):
        """Handle the logic for logging a problem and saving a video clip of a specified duration."""

        relative_clip_path = 'problemClip/' + video_clip_path
        absolute_clip_path = PROBLEM_CLIPS_PATH + video_clip_path
        # Get the frame rate of the video
        fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        total_frames = fps * clip_duration  # Total frames to capture for the clip duration

        # Initialize the video writer with the correct dimensions and format
        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(absolute_clip_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (frame_width, frame_height))

        # Capture and write the frames for the specified duration
        for _ in range(total_frames):
            success, frame = self.video_capture.read()
            if not success:
                break
            out.write(frame)

        out.release()



        # Log the detected issue in the database
        log_detected_issue(
            self.database_connection,
            self.database_cursor,
            problem_name,
            problem_description,
            relative_clip_path,
            table_number,
            self.display_detection_logs
        )

    def monitor_fights(self, img, fights):
        """Monitor and log detected fights."""
        if fights:
            for (fx1, fy1, fx2, fy2) in fights:
                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), CLASS_COLORS['fight'], 1, 2)
                issue_description = f"Fight detected at [{fx1}, {fy1}, {fx2}, {fy2}]."
                video_clip_path = f"fight_detected.mp4"

                self.handle_problem_logging(
                    problem_name="Fight Detected",
                    problem_description=issue_description,
                    video_clip_path=video_clip_path,
                    table_number=None
                )

    def resize_video_frame(self, frame, target_height=720):
        """Resize the video frame to a specific height while maintaining the aspect ratio."""
        height, width = frame.shape[:2]
        scaling_factor = target_height / height
        new_width = int(width * scaling_factor)
        return cv2.resize(frame, (new_width, target_height))

    def run(self):
        """Main loop to process video frames."""
        paused = False
        delay = int(1000 / 30 / self.playback_speed)

        while True:
            if not paused:
                success, img = self.video_capture.read()
                if not success:
                    break

                img = self.resize_video_frame(img)
                tables, people, waiters, money, food_items, fights = process_video_frame(
                    img,
                    self.object_detection_model,
                    self.display_detection_logs,
                    self.assign_table_identifier,
                    detect_and_log_faces,
                    self.known_face_encodings,
                    self.waiter_names,
                    self.database_cursor
                )

                self.monitor_table_and_food_status(img, tables, people, food_items)
                self.monitor_fights(img, fights)

                cv2.imshow("Restaurant Monitoring", img)

            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break

            if key == ord('p'):
                paused = not paused
                if paused:
                    print("Paused. Press 'p' to continue.")

        self.video_capture.release()
        cv2.destroyAllWindows()
        self.database_connection.close()
