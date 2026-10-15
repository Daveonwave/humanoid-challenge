"""Extract board-relative marker positions; never interpret them as fingertip poses.
Requires OpenCV with aruco and NumPy. Outputs are offline analysis, not robot actions.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def smooth(values, median_window=5, mean_window=7):
    padded = np.pad(values, ((median_window//2, median_window//2), (0, 0)), mode='edge')
    med = np.array([np.median(padded[i:i+median_window], axis=0) for i in range(len(values))])
    padded = np.pad(med, ((mean_window//2, mean_window//2), (0, 0)), mode='edge')
    return np.array([padded[i:i+mean_window].mean(axis=0) for i in range(len(values))])


def run(root, output, config_path=None):
    config = json.loads((config_path or root/'configs/pilot_01.json').read_text())
    calibration = np.load(root/config['calibration'])
    K, dist = calibration['camera_matrix'], calibration['distortion']
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), .030, .022, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    half = .040/2
    marker_points = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]], np.float32)
    cap = cv2.VideoCapture(str(root/config['video']))
    
    if not cap.isOpened(): 
        raise RuntimeError('Cannot open pilot video')
    
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Invalid video FPS')
    rows = []; decoded = 0; detected = 0; raw_times = []
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok: 
                break
            index = decoded; decoded += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            raw_times.append(timestamp)
            if tuple(calibration['image_size']) != (frame.shape[1], frame.shape[0]):
                raise ValueError('Decoded orientation/resolution differs from calibration')
            
            cc, ci, mc, mi = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ci is None or len(ci)<12 or mi is None or 49 not in mi.flatten(): 
                continue
            
            detected += 1
            obj, img = board.matchImagePoints(cc, ci)
            okb, rb, tb = cv2.solvePnP(obj, img, K, dist)
            marker = np.asarray(mc[list(mi.flatten()).index(49)], np.float32).reshape(4,2)
            
            okm, rm, tm = cv2.solvePnP(marker_points, marker, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not (okb and okm): 
                continue
            
            Rb = cv2.Rodrigues(rb)[0]
            position = (Rb.T @ (tm-tb)).ravel()
            
            # Define height positive toward the camera-facing side of the board.
            normal_sign = 1.0 if float(Rb[:,2] @ (-tb.ravel())) > 0 else -1.0
            position[2] *= normal_sign
            mp = cv2.projectPoints(marker_points, rm, tm, K, dist)[0].reshape(-1,2)
            bp = cv2.projectPoints(obj, rb, tb, K, dist)[0].reshape(-1,2)
            me = float(np.sqrt(np.mean(np.sum((mp-marker)**2, axis=1))))
            be = float(np.sqrt(np.mean(np.sum((bp-img.reshape(-1,2))**2, axis=1))))
            rows.append([index, timestamp, *position, me, be])
            
            if index % 90 == 0: 
                print(f'Processed frame {index}', flush=True)
    finally:
        cap.release()
    
    arr = np.asarray(rows) 
    if len(arr)<15: 
        raise RuntimeError('Too few detected frames')
    
    timestamps_ok = len(raw_times)>1 and np.all(np.diff(raw_times)>0)
    if not timestamps_ok: 
        arr[:,1] = arr[:,0]/fps
    
    arr[:,1] -= raw_times[0] if timestamps_ok else 0
    valid = (arr[:,5]<config['max_reprojection_px']) & (arr[:,6]<config['max_reprojection_px'])
    good = arr[valid]
    
    if len(good)<15: 
        raise RuntimeError('Too few quality-filtered frames')
    
    if np.max(np.diff(good[:,1])) > .25: 
        raise RuntimeError('Tracking gap >0.25 s; inspect before interpolation')
    
    ts = good[:,1]
    uniform = np.arange(ts[0], ts[-1]+1e-9, 1/config['output_hz'])
    positions = np.column_stack([np.interp(uniform,ts,good[:,j]) for j in (2,3,4)])
    filtered = smooth(positions)
    output.mkdir(parents=True, exist_ok=True)
    
    def save(name, header, data):
        with (output/name).open('w', newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(data)
    
    save('marker_raw.csv', ['frame','seconds','x_m','y_m','height_m','marker_error_px','board_error_px'], arr)
    save('marker_smoothed.csv', ['seconds','x_m','y_m','height_m'], np.column_stack([uniform,filtered]))
    
    report = dict(opencv=cv2.__version__, fps=fps, decoded_frames=decoded,
        paired_detections=detected, valid_poses=len(arr), quality_accepted=len(good),
        timestamps='video_POS_MSEC' if timestamps_ok else 'frame_index_div_fps',
        median_marker_error_px=float(np.median(arr[:,5])),
        median_board_error_px=float(np.median(arr[:,6])),
        max_adjacent_raw_position_change_m=float(np.max(np.linalg.norm(np.diff(good[:,2:5],axis=0),axis=1))),
        source_video=config['video'], source_calibration=config['calibration'],
        interpretation='Marker center in board frame; positive height faces camera. Not a measured grasp point.',
        limitations='Planar pose ambiguity and intrinsics error are not excluded by low reprojection error. Offline smoothing can shift phase boundaries.')
    (output/'tracking_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output',type=Path)
    parser.add_argument('--config', type=Path)
    args=parser.parse_args();run(args.project,args.output or args.project/'data/processed/pilot_01', args.config)
