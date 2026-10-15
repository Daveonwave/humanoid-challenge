# Pilot 01 trajectory and first hybrid replay

## Verified locally

- Source: `data/raw/pilot_pick_place_01.mov`.
- 434 decoded frames, all with marker 49 and enough board corners for pose estimation.
- All 434 passed the provisional 2-pixel reprojection threshold for both poses.
- Median reprojection errors: marker 0.261 px; board 0.150 px.
- Maximum adjacent raw position change: 12.2 mm. Detection completeness does not rule out pose jitter or planar ambiguity.
- Dimensions match the portrait calibration (1080 x 1920).
- Timestamp source: decoded video timestamps, not assumed 30 fps. OpenCV reports average fps 29.087.
- Python syntax checked for the processing script and notebook cells. LIBERO physics rollout has NOT been executed locally.

## Files

- `scripts/process_pilot.py`: extract, quality-filter, resample at 20 Hz, median-filter and moving-average marker positions.
- `configs/pilot_01.json`: source paths and provisional manually annotated phases.
- `data/processed/pilot_01/marker_raw.csv`: raw poses and errors.
- `data/processed/pilot_01/marker_smoothed.csv`: offline smoothed trajectory.
- `data/processed/pilot_01/tracking_report.json`: quality report.
- `notebooks/03_pilot_retargeting.ipynb`: inspection and experimental hybrid rollout in Colab.

## Honest scope of the method

The hand marker is not the grasp point. There is no calibrated marker-to-finger rigid transform, and human finger articulation would make a constant transform approximate anyway. This first experiment transfers path shape between robot-expert anchors. The expert supplies approach/grasp actions, endpoints and orientation; the human clip supplies transfer timing and deviations from an endpoint-to-endpoint path.

This is a hybrid bootstrap, not direct imitation of raw human actions and not yet VLA training. Do not report a successful run as evidence that human data improves performance. A subsequent straight-line ablation with identical anchors and timing is needed.

The 5.0–11.5 s transfer interval and approximately 12 s release are provisional visual annotations. Review the video and plots. The notebook's robot anchor heuristic must also be reviewed; gripper command timing alone does not establish a successful grasp.

## Run

Open notebook 03 in a Colab Linux runtime. It loads already-processed data, so there is no need to rerun tracking first. Run sections 1–3 and inspect the motion and anchor images before sections 4–5. Save the plot, video and success report. A failed rollout is a diagnostic; do not export it as expert training data.

To regenerate tracking locally or in a separate computer-vision runtime:

```sh
python scripts/process_pilot.py --project /path/to/humanoid-challenge
```

Requires NumPy and OpenCV with aruco (local validation used OpenCV 5.0.0). Outputs use board x/y and camera-facing height, in metres. Changing camera optics, crop, resolution or orientation invalidates assumptions unless handled explicitly.
