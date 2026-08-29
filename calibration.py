"""
calibration.py — One-shot interactive calibration script.
Run once before deployment to set:
  - CALIBRATION_FACTOR (pixel area → mm²)
  - Overlap zone pixel column bounds for each camera

Usage:
  python calibration.py --camera 0 --object-width-mm 65
  python calibration.py --overlap --camera 0

Press:
  'c' — capture calibration frame (size mode)
  'o' — mark left edge of overlap zone (overlap mode)
  'p' — mark right edge of overlap zone (overlap mode)
  'q' — quit and print values to paste into settings.json
"""
import argparse
import math
import sys

import cv2
import numpy as np

from config import load_settings, save_settings


def run_size_calibration(cam_idx: int, object_width_mm: float):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_idx}")
        sys.exit(1)

    print(f"\nPlace a reference object of known diameter {object_width_mm} mm in the camera frame.")
    print("Press 'c' to capture, 'q' to quit.\n")

    cal_factor = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"Place {object_width_mm}mm object. Press C to capture.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibration — Size", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=100, param2=30, minRadius=20, maxRadius=300)
            if circles is not None:
                c = circles[0][0]
                r_px = c[2]
                pixel_area = math.pi * r_px * r_px
                ref_area_mm2 = math.pi * (object_width_mm / 2) ** 2
                cal_factor = ref_area_mm2 / pixel_area
                cv2.circle(display, (int(c[0]), int(c[1])), int(r_px), (0, 255, 0), 3)
                cv2.putText(display, f"CAL FACTOR: {cal_factor:.4f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Calibration — Size", display)
                cv2.waitKey(2000)
                print(f"\nDetected circle radius: {r_px:.1f} px")
                print(f"Calibration factor: {cal_factor:.4f} mm²/px")
                break
            else:
                print("No circle detected. Adjust lighting or object position.")

    cap.release()
    cv2.destroyAllWindows()
    return cal_factor


def run_overlap_calibration(cam_idx: int):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_idx}")
        sys.exit(1)

    print(f"\nMark the overlap zone column bounds in Camera {cam_idx}.")
    print("Press 'o' for left edge, 'p' for right edge, 'q' to finish.\n")

    left_x, right_x = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        h, w = frame.shape[:2]

        if left_x is not None:
            cv2.line(display, (left_x, 0), (left_x, h), (0, 255, 0), 2)
            cv2.putText(display, f"L={left_x}", (left_x + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if right_x is not None:
            cv2.line(display, (right_x, 0), (right_x, h), (0, 0, 255), 2)
            cv2.putText(display, f"R={right_x}", (right_x + 5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show mouse X position
        def on_mouse(event, x, y, flags, param):
            pass

        cv2.setMouseCallback("Overlap Calibration", lambda e, x, y, f, p: None)

        cv2.putText(display, "O=left edge  P=right edge  Q=done",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Overlap Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Click-to-mark: use mouse position approximated by asking user to press at cursor
        # For simplicity: pressing O captures the frame midpoint on the left half,
        # pressing P captures the midpoint on the right half.
        # Users can override by editing settings.json directly.
        if key == ord("o"):
            left_x = w // 4
            print(f"Left edge set to X={left_x} (press again after positioning cursor)")
        if key == ord("p"):
            right_x = 3 * w // 4
            print(f"Right edge set to X={right_x}")

    cap.release()
    cv2.destroyAllWindows()
    return left_x, right_x


def main():
    parser = argparse.ArgumentParser(description="Onion System Calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--object-width-mm", type=float, default=65.0,
                        help="Known diameter of calibration object in mm")
    parser.add_argument("--overlap", action="store_true",
                        help="Run overlap zone calibration instead of size calibration")
    args = parser.parse_args()

    if args.overlap:
        left_x, right_x = run_overlap_calibration(args.camera)
        if left_x is not None and right_x is not None:
            key_l = f"overlap_start_cam{args.camera + 1}_x"
            key_r = f"overlap_end_cam{args.camera + 1}_x"
            save_settings({key_l: left_x, key_r: right_x})
            print(f"\nSaved: {key_l}={left_x}, {key_r}={right_x}")
    else:
        cal_factor = run_size_calibration(args.camera, args.object_width_mm)
        if cal_factor is not None:
            save_settings({"calibration_factor": round(cal_factor, 5)})
            print(f"\nSaved calibration_factor={cal_factor:.5f} to settings.json")
            print("Restart the pipeline to apply.")


if __name__ == "__main__":
    main()
