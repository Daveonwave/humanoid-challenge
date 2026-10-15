import cv2
from cv2 import aruco

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "markers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("OpenCV version:", cv2.__version__)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Physical dimensions in metres.
board = aruco.CharucoBoard(
    (5, 7),
    0.030,  # 30 mm squares
    0.022,  # 22 mm internal markers
    aruco_dict,
)

# Preserve the board's 5:7 proportions.
board_image = board.generateImage((1500, 2100))

# This returns an image directly. Size here is in pixels.
marker_image = aruco.generateImageMarker(aruco_dict, 49, 600)

# White margins help detection at the outer edges.
board_image = cv2.copyMakeBorder(
    board_image,
    100,
    100,
    100,
    100,
    cv2.BORDER_CONSTANT,
    value=255,
)
marker_image = cv2.copyMakeBorder(
    marker_image,
    75,
    75,
    75,
    75,
    cv2.BORDER_CONSTANT,
    value=255,
)

assert cv2.imwrite(str(OUTPUT_DIR / "charuco_board.png"), board_image)
assert cv2.imwrite(str(OUTPUT_DIR / "aruco_marker_49.png"), marker_image)

print("Saved both images.")
