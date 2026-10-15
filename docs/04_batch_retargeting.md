# Batch human-video retargeting

## Objective and boundaries
Generate reviewed robot episodes from 10–20 human recordings in the same seeded LIBERO scene. Human motion determines transfer shape; release-v3 supplies engineered approach, grasp, orientation, contact-based placement and timing. This is dataset generation, not VLA inference. The phone videos are not VLA image inputs.

Use `08_batch_human_retargeting.ipynb` in the working Colab environment used for notebook 05. It invokes scripts from Drive. The controller is a static extraction of notebook 05, not dynamically executed notebook content. Notebook 05 remains an independent pilot; later controller changes should be made deliberately in the script.

## Recording and registration
Save new recordings as `data/raw/demo_001.mov`, etc. Keep the pilot untouched. Record with the calibrated lens/resolution/orientation and board/marker dimensions (5×7 board, 30 mm squares, 22 mm board markers, hand marker ID 49 with 40 mm black square). Maintain visibility of board corners and hand marker. Each video contains one complete transfer.

Register only the intended new recordings. The notebook selects `demo_*` filenames, so it will not ingest calibration videos. Configuration is stored per video, and `configs/retarget_batch.json` records progress. Use the appropriate calibration for the new recording settings; changing table alone does not change camera intrinsics.

## Processing
1. **Track** extracts and smooths board-relative marker motion. Existing tracker quality checks reject resolution mismatch, insufficient detections and internal gaps over 0.25 s. This does not rule out planar pose ambiguity; inspect the resulting transfer/replay.
2. **Review events** fills six events in `data/processed/<demo>/human_events.json`. Use zero-based decoded frames and video times. The annotation helper in notebook 08 displays a chosen decoded frame and records its timestamp. A frame-rate fallback is used only if video timestamps are unusable. Mark closure start, secure grasp, lift-off, placement contact, opening start and complete release. Click through ambiguous boundaries; marker trajectories alone cannot prove contact. Then run `approve-events`.
3. **Retarget** launches a separate process per video, using the same seed and release-v3 controller settings. Each run writes `episode.h5`, `replay.mp4`, `mapped_path.npz`, and `report.json` on local Colab disk, then copies completed attempts to Drive. During execution inspect `/content/humanoid_retarget_runs/logs`; after copying, logs are in `outputs/batch_logs`. The runner continues to the next demo after a failure. Detailed progress is in `outputs/batch_logs`; simulation can take several minutes per episode.
4. **Review replay** confirms stable grasp, transfer, placement and full release. Approve or reject explicitly. Passing simulator checks alone does not make data training-eligible.
5. **Export list** writes `data/robot/approved_batch.json`, containing only current approved episodes. This is an input inventory, not a LeRobot dataset conversion. Notebook 09 exports this approved list as a multi-episode LeRobot dataset.

Rerunning track/retarget skips unchanged completed jobs. Use `--retry` to retry unchanged failures. Changes to video, calibration, tracking configuration, tracker, trajectory, events, or controller invalidate the corresponding cached stage. Changed annotations require another `approve-events`. Old simulation runs are retained. Do not run two batch writers concurrently.

## CLI
Run with the project's Python environment:

```bash
python scripts/batch_retarget.py --project /path/to/humanoid-challenge status
python scripts/batch_retarget.py --project /path/to/humanoid-challenge track
python scripts/batch_retarget.py --project /path/to/humanoid-challenge approve-events --demo demo_001
python scripts/batch_retarget.py --project /path/to/humanoid-challenge retarget
python scripts/batch_retarget.py --project /path/to/humanoid-challenge review --demo demo_001 --decision approve --note "Stable grasp, placement and complete release visible."
python scripts/batch_retarget.py --project /path/to/humanoid-challenge export-list
```

Track requires NumPy/OpenCV with ArUco. Retarget requires the working notebook-05 LIBERO/robosuite/MuJoCo environment, scipy, pandas, h5py and imageio with ffmpeg. Do not upgrade that environment merely to run this launcher. EGL rendering is configured for Colab.

## Validation
Standard-library tests cover event ordering/coverage, missing annotations and stale tracking detection. Python syntax is checked. Full simulator validation must be run in Colab; a successful pilot does not guarantee all new paths are physically feasible. Start the batch on one new episode, inspect its replay, then let the remaining reviewed inputs run unattended.
