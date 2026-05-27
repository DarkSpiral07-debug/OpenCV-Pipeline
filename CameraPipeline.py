import cv2
import numpy as np
import time
import os
class CameraPipeline:
    def __init__(self):
        os.makedirs("outputs", exist_ok=True)   
        self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.orb = cv2.ORB_create(nfeatures=120)
        self.prev_time = 0
    def resize_frame(self, frame):
        resized = cv2.resize(frame, (640, 480))
        return resized
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        filtered = cv2.medianBlur(blurred, 5)
        return filtered
    def normalize_frame(self, frame):
        normalized = frame / 255.0
        return normalized
    def detect_edges(self, frame):
        edges = cv2.Canny(frame, 100, 200)
        return edges
    def detect_corners(self, frame, display_frame):
        corners = cv2.goodFeaturesToTrack(frame, 80, 0.01, 10)
        output = display_frame.copy()
        if corners is not None:
            corners = np.int32(corners)
            for point in corners:
                x, y = point.ravel()
                cv2.circle(output, (x, y), 4, (0, 0, 255), -1)
        return output
    def extract_orb(self, frame):
        keypoints = self.orb.detect(frame, None)
        orb_frame = cv2.drawKeypoints(
            frame,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=0
        )
        return orb_frame
    def calculate_fps(self):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        return int(fps)
    def run_pipeline(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                break
            original = frame.copy()
            resized = self.resize_frame(frame)
            preprocessed = self.preprocess_frame(resized)
            normalized = self.normalize_frame(preprocessed)
            edges = self.detect_edges(preprocessed)
            corners = self.detect_corners(preprocessed, original)
            orb_features = self.extract_orb(preprocessed)
            fps = self.calculate_fps()
            cv2.putText(
                original,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("Original Feed", original)
            cv2.imshow("Preprocessed Frame", preprocessed)
            cv2.imshow("Normalized Frame", normalized)
            cv2.imshow("Edge Features", edges)
            cv2.imshow("Corner Features", corners)
            cv2.imshow("ORB Keypoints", orb_features)
            key = cv2.waitKey(1)
            if key == ord('s'):
                filename = f"outputs/capture_{int(time.time())}.png"
                cv2.imwrite(filename, original)
                print(f"Saved: {filename}")
            if key == ord('q'):
                break
        self.camera.release()
        cv2.destroyAllWindows()
def main():
    system = CameraPipeline()
    system.run_pipeline()
if __name__ == "__main__":
    main()