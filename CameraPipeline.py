import cv2
import numpy as np
import os
import time
from collections import deque


# ─────────────────────────────────────────────────────────────
#  CONFIG  –  tune all parameters here without touching logic
# ─────────────────────────────────────────────────────────────
class Config:
    # ── Input / Output ──────────────────────────────────────
    INPUT_FILE  = os.path.join("input",   "lane_video.mp4")
    OUTPUT_FILE = os.path.join("outputs", "lane_output.mp4")

    # ── HLS White Mask ──────────────────────────────────────
    WHITE_L_LOW  = 180
    WHITE_L_HIGH = 255
    WHITE_S_LOW  = 0

    # ── HLS Yellow Mask ─────────────────────────────────────
    YELLOW_H_LOW  = 15
    YELLOW_H_HIGH = 35
    YELLOW_L_LOW  = 80
    YELLOW_S_LOW  = 100

    # ── Gaussian Blur ────────────────────────────────────────
    BLUR_KERNEL = (5, 5)

    # ── Canny Edge Detection ─────────────────────────────────
    CANNY_LOW  = 50
    CANNY_HIGH = 150

    # ── ROI Trapezoid (fractions of frame size) ───────────────
    ROI_BOTTOM_LEFT_X  = 0.05
    ROI_BOTTOM_RIGHT_X = 0.95
    ROI_TOP_LEFT_X     = 0.42
    ROI_TOP_RIGHT_X    = 0.58
    ROI_TOP_Y          = 0.60
    ROI_BOTTOM_Y       = 1.00

    # ── Hough Transform ──────────────────────────────────────
    HOUGH_RHO          = 1
    HOUGH_THETA        = np.pi / 180
    HOUGH_THRESHOLD    = 50
    HOUGH_MIN_LENGTH   = 50
    HOUGH_MAX_GAP      = 150

    # ── Line Filtering ───────────────────────────────────────
    MIN_SLOPE_ABS      = 0.40
    MAX_SLOPE_ABS      = 2.50
    EDGE_MARGIN        = 0.10

    # ── Temporal Smoothing ───────────────────────────────────
    SMOOTH_WINDOW      = 8

    # ── Offset Thresholds (pixels) ───────────────────────────
    DRIFT_THRESHOLD    = 40

    # ── Visualization ────────────────────────────────────────
    COLOR_LEFT_LANE    = (255,   0, 200)
    COLOR_RIGHT_LANE   = (255,   0, 200)
    COLOR_CORRIDOR     = (  0, 255,   0)
    COLOR_VEHICLE_CTR  = (  0, 255, 255)
    COLOR_LANE_CTR     = (255, 255,   0)
    COLOR_TEXT         = (255, 255, 255)
    LANE_LINE_THICK    = 8
    CENTER_LINE_THICK  = 3
    CORRIDOR_ALPHA     = 0.25


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
class LaneDetectionPipeline:

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()

        base = os.path.dirname(os.path.abspath(__file__))
        self.input_path  = os.path.join(base, self.cfg.INPUT_FILE)
        self.output_path = os.path.join(base, self.cfg.OUTPUT_FILE)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        print(f"Loading: {self.input_path}")
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.input_path}")

        self.W   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.W, self.H)
        )

        n = self.cfg.SMOOTH_WINDOW
        self.left_buf  = deque(maxlen=n)
        self.right_buf = deque(maxlen=n)

        self._prev_t = time.time()
        print("Pipeline ready.")

    # ──────────────────────────────────────────────────────────
    #  STAGE 1 – PREPROCESSING
    # ──────────────────────────────────────────────────────────
    def preprocess(self, frame):
        c = self.cfg
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        white_mask = cv2.inRange(
            hls,
            np.array([0,   c.WHITE_L_LOW,  c.WHITE_S_LOW]),
            np.array([180, c.WHITE_L_HIGH, 255])
        )

        yellow_mask = cv2.inRange(
            hls,
            np.array([c.YELLOW_H_LOW,  c.YELLOW_L_LOW, c.YELLOW_S_LOW]),
            np.array([c.YELLOW_H_HIGH, 255,             255])
        )

        combined = cv2.bitwise_or(white_mask, yellow_mask)

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        blur  = cv2.GaussianBlur(combined, c.BLUR_KERNEL, 0)
        edges = cv2.Canny(blur, c.CANNY_LOW, c.CANNY_HIGH)

        return edges, combined

    # ──────────────────────────────────────────────────────────
    #  STAGE 2 – REGION OF INTEREST
    # ──────────────────────────────────────────────────────────
    def region_of_interest(self, edges):
        c     = self.cfg
        H, W  = edges.shape
        mask  = np.zeros_like(edges)

        pts = np.array([[
            (int(W * c.ROI_BOTTOM_LEFT_X),  int(H * c.ROI_BOTTOM_Y)),
            (int(W * c.ROI_BOTTOM_RIGHT_X), int(H * c.ROI_BOTTOM_Y)),
            (int(W * c.ROI_TOP_RIGHT_X),    int(H * c.ROI_TOP_Y)),
            (int(W * c.ROI_TOP_LEFT_X),     int(H * c.ROI_TOP_Y)),
        ]], dtype=np.int32)

        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(edges, mask), mask

    # ──────────────────────────────────────────────────────────
    #  STAGE 3 – HOUGH TRANSFORM + LINE FILTERING
    # ──────────────────────────────────────────────────────────
    def detect_lines(self, roi):
        c = self.cfg
        raw = cv2.HoughLinesP(
            roi,
            c.HOUGH_RHO,
            c.HOUGH_THETA,
            threshold=c.HOUGH_THRESHOLD,
            minLineLength=c.HOUGH_MIN_LENGTH,
            maxLineGap=c.HOUGH_MAX_GAP
        )
        if raw is None:
            return None

        filtered  = []
        margin_px = int(self.W * c.EDGE_MARGIN)

        for seg in raw:
            x1, y1, x2, y2 = seg.reshape(4)
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if not (c.MIN_SLOPE_ABS <= abs(slope) <= c.MAX_SLOPE_ABS):
                continue

            mid_x = (x1 + x2) / 2
            if mid_x < margin_px or mid_x > (self.W - margin_px):
                continue

            filtered.append(seg)

        return np.array(filtered) if filtered else None

    # ──────────────────────────────────────────────────────────
    #  STAGE 4 – LINE AVERAGING
    # ──────────────────────────────────────────────────────────
    def average_lines(self, lines):
        if lines is None:
            return None, None

        left_pts, right_pts = [], []
        cx = self.W / 2

        for seg in lines:
            x1, y1, x2, y2 = seg.reshape(4)
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            mid_x = (x1 + x2) / 2

            if slope < 0 and mid_x < cx:
                left_pts.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and mid_x >= cx:
                right_pts.extend([(x1, y1), (x2, y2)])

        return self._fit_line(left_pts), self._fit_line(right_pts)

    def _fit_line(self, pts):
        if len(pts) < 2:
            return None
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        try:
            slope, intercept = np.polyfit(ys, xs, 1)
        except np.linalg.LinAlgError:
            return None

        y1 = self.H
        y2 = int(self.H * self.cfg.ROI_TOP_Y)
        x1 = int(slope * y1 + intercept)
        x2 = int(slope * y2 + intercept)
        return np.array([x1, y1, x2, y2])

    # ──────────────────────────────────────────────────────────
    #  STAGE 5 – TEMPORAL SMOOTHING
    # ──────────────────────────────────────────────────────────
    def smooth_lines(self, left_line, right_line):
        if left_line  is not None: self.left_buf.append(left_line)
        if right_line is not None: self.right_buf.append(right_line)

        smoothed_left  = np.mean(self.left_buf,  axis=0).astype(int) if self.left_buf  else None
        smoothed_right = np.mean(self.right_buf, axis=0).astype(int) if self.right_buf else None
        return smoothed_left, smoothed_right

    # ──────────────────────────────────────────────────────────
    #  STAGE 6 – VISUALIZATION
    # ──────────────────────────────────────────────────────────
    def draw_output(self, frame, left_line, right_line, fps):
        c      = self.cfg
        output = frame.copy()
        H, W   = self.H, self.W

        vehicle_cx = W // 2

        # vehicle center reference line (cyan, dashed)
        self._draw_dashed_line(
            output,
            (vehicle_cx, H), (vehicle_cx, int(H * c.ROI_TOP_Y)),
            c.COLOR_VEHICLE_CTR, c.CENTER_LINE_THICK
        )

        # lane lines
        if left_line is not None:
            cv2.line(output,
                     (left_line[0],  left_line[1]),
                     (left_line[2],  left_line[3]),
                     c.COLOR_LEFT_LANE, c.LANE_LINE_THICK)

        if right_line is not None:
            cv2.line(output,
                     (right_line[0],  right_line[1]),
                     (right_line[2],  right_line[3]),
                     c.COLOR_RIGHT_LANE, c.LANE_LINE_THICK)

        # handle missing lane by estimating from the visible one
        lane_width_est = int(self.W * 0.35)

        if left_line is not None and right_line is None:
            right_line = left_line.copy()
            right_line[0] += lane_width_est
            right_line[2] += lane_width_est
        elif right_line is not None and left_line is None:
            left_line = right_line.copy()
            left_line[0] -= lane_width_est
            left_line[2] -= lane_width_est

        if left_line is not None and right_line is not None:
            # green filled corridor
            overlay = frame.copy()
            poly = np.array([[
                [left_line[0],  left_line[1]],
                [left_line[2],  left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]],
            ]], dtype=np.int32)
            cv2.fillPoly(overlay, poly, c.COLOR_CORRIDOR)
            output = cv2.addWeighted(
                output, 1 - c.CORRIDOR_ALPHA,
                overlay, c.CORRIDOR_ALPHA, 0
            )

            # lane center line (yellow)
            lane_cx = (left_line[0] + right_line[0]) // 2
            cv2.line(output,
                     (lane_cx, H),
                     (lane_cx, int(H * c.ROI_TOP_Y)),
                     c.COLOR_LANE_CTR, c.CENTER_LINE_THICK)

            # offset and status
            offset       = vehicle_cx - lane_cx
            status       = self._lane_status(offset)
            status_color = (0, 200, 0) if status == "CENTERED" else (0, 60, 255)
            self._draw_hud(output, fps, offset, status, status_color)
        else:
            self._draw_hud(output, fps, offset=None, status="NO LANE", status_color=(0, 60, 255))

        return output

    def _lane_status(self, offset):
        th = self.cfg.DRIFT_THRESHOLD
        if   offset >  th: return "DRIFT LEFT"
        elif offset < -th: return "DRIFT RIGHT"
        else:              return "CENTERED"

    def _draw_hud(self, img, fps, offset, status, status_color):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thick = 2
        pad   = 10

        lines = [f"FPS: {fps}"]
        if offset is not None:
            lines.append(f"Offset: {offset:+d}px")
        lines.append(f"Status: {status}")

        for i, text in enumerate(lines):
            y = pad + (i + 1) * 30
            (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(img,
                          (pad - 4, y - th_ - 4),
                          (pad + tw + 4, y + 4),
                          (0, 0, 0), -1)
            color = status_color if "Status" in text else self.cfg.COLOR_TEXT
            cv2.putText(img, text, (pad, y), font, scale, color, thick)

    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=15, gap_len=10):
        x1, y1 = pt1
        x2, y2 = pt2
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist
        drawn   = 0
        drawing = True
        while drawn < dist:
            seg = dash_len if drawing else gap_len
            seg = min(seg, dist - drawn)
            ex  = int(x1 + dx * (drawn + seg))
            ey  = int(y1 + dy * (drawn + seg))
            if drawing:
                cv2.line(img,
                         (int(x1 + dx * drawn), int(y1 + dy * drawn)),
                         (ex, ey), color, thickness)
            drawn  += seg
            drawing = not drawing

    # ──────────────────────────────────────────────────────────
    #  FPS COUNTER
    # ──────────────────────────────────────────────────────────
    def _calc_fps(self):
        now = time.time()
        fps = 1.0 / max(now - self._prev_t, 1e-6)
        self._prev_t = now
        return int(fps)

    # ──────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────────────────────
    def run(self):
        print("Processing – press Q to quit early.")
        frame_no = 0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame_no += 1

            edges, color_mask         = self.preprocess(frame)
            roi_edges, _              = self.region_of_interest(edges)
            raw_lines                 = self.detect_lines(roi_edges)
            left_raw, right_raw       = self.average_lines(raw_lines)
            left_smooth, right_smooth = self.smooth_lines(left_raw, right_raw)

            # debug print every 10 frames
            if frame_no % 10 == 0:
                print(
                    "raw:", raw_lines is not None,
                    "| left:", left_raw is not None,
                    "| right:", right_raw is not None
                )

            fps    = self._calc_fps()
            output = self.draw_output(frame, left_smooth, right_smooth, fps)

            self.writer.write(output)
            cv2.imshow("1 - Original",         frame)
            cv2.imshow("2 - White/Yellow Mask", color_mask)
            cv2.imshow("3 - Edge Detection",    edges)
            cv2.imshow("4 - ROI",               roi_edges)
            cv2.imshow("5 - Lane Detection",    output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        print(f"Done. {frame_no} frames written to:\n  {self.output_path}")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    pipeline = LaneDetectionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()