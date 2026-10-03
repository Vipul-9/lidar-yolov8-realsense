"""
LiDAR-based object detection and counting.

YOLOv8 runs on the RealSense L515 RGB stream (live or a recorded .bag file).
Each detection is tagged with its distance from the aligned depth frame, and
objects are counted per class. Results are shown with OpenCV; the 3D point
cloud of the scene can be viewed with Open3D.

Usage:
    python main.py --live                        # live L515 stream
    python main.py --bag recordings/scene.bag    # recorded stream
    python main.py --video clip.mp4              # plain video / phone footage (no depth)
    python main.py --live --pointcloud           # also show the 3D point cloud (press 'p')
"""

import argparse
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class RealSenseSource:
    """Live L515 stream or recorded .bag file, with depth aligned to colour."""

    def __init__(self, bag_path=None, width=1280, height=720, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed: pip install pyrealsense2")
        self.pipeline = rs.pipeline()
        config = rs.config()
        if bag_path:
            config.enable_device_from_file(bag_path, repeat_playback=False)
        else:
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        if bag_path:
            profile.get_device().as_playback().set_real_time(False)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        self.last_depth = None
        self.last_color = None

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return None, None  # end of recording / no frames
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None, None
        self.last_depth, self.last_color = depth, color
        img = np.asanyarray(color.get_data())
        if img.shape[2] == 3 and color.get_profile().format() == rs.format.rgb8:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return img, depth_m

    def point_cloud(self):
        """Return (points Nx3, colours Nx3 in 0-1) for the latest frame."""
        if self.last_depth is None:
            return None, None
        self.pc.map_to(self.last_color)
        points = self.pc.calculate(self.last_depth)
        xyz = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        uv = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        img = np.asanyarray(self.last_color.get_data())
        h, w = img.shape[:2]
        u = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        v = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
        rgb = img[v, u][:, ::-1] / 255.0  # BGR -> RGB
        keep = xyz[:, 2] > 0
        return xyz[keep], rgb[keep]

    def release(self):
        self.pipeline.stop()


class VideoSource:
    """Plain video file or webcam (no depth)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(0 if path == "0" else path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

    def read(self):
        ok, img = self.cap.read()
        return (img, None) if ok else (None, None)

    def point_cloud(self):
        return None, None

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #
def box_distance(depth_m, x1, y1, x2, y2):
    """Median depth (m) of the central half of a box; robust to background."""
    if depth_m is None:
        return None
    cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
    cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
    patch = depth_m[cy1:cy2, cx1:cx2]
    valid = patch[(patch > 0.25) & (patch < 9.0)]  # L515 operating range
    return float(np.median(valid)) if valid.size else None


def colour_for(cls_id):
    rng = np.random.default_rng(cls_id)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def annotate(img, depth_m, result, names):
    counts = Counter()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id, conf = int(box.cls[0]), float(box.conf[0])
        name = names[cls_id]
        counts[name] += 1
        dist = box_distance(depth_m, x1, y1, x2, y2)
        label = f"{name} {conf:.2f}" + (f" | {dist:.2f} m" if dist else "")
        col = colour_for(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Count overlay
    y = 30
    cv2.putText(img, f"People Count: {counts.get('person', 0)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    for name, n in sorted(counts.items()):
        if name == "person":
            continue
        y += 26
        cv2.putText(img, f"{name}: {n}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return counts


def show_point_cloud(source):
    import open3d as o3d
    xyz, rgb = source.point_cloud()
    if xyz is None:
        print("Point cloud needs a RealSense source (--live or --bag).")
        return
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd = pcd.voxel_down_sample(0.01)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # camera -> viewer frame
    o3d.visualization.draw_geometries([pcd], window_name="L515 point cloud")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="YOLOv8 + RealSense L515 detection and counting")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true", help="live L515 stream")
    src.add_argument("--bag", help="recorded RealSense .bag file")
    src.add_argument("--video", help="video file, or 0 for webcam")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--pointcloud", action="store_true", help="enable 'p' key for 3D view")
    ap.add_argument("--save", help="save annotated output to this .mp4")
    args = ap.parse_args()

    model = YOLO(args.model)
    source = VideoSource(args.video) if args.video else RealSenseSource(bag_path=args.bag)
    writer = None

    try:
        while True:
            img, depth_m = source.read()
            if img is None:
                break
            result = model(img, conf=args.conf, verbose=False)[0]
            annotate(img, depth_m, result, model.names)

            if args.save:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(img)

            cv2.imshow("LiDAR object detection", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p") and args.pointcloud:
                show_point_cloud(source)
    finally:
        source.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
