# Training SmolVLA with human demonstrations

Can a robot learn to move a bowl from videos recorded at home? This project follows that idea from iPhone recordings to robot demonstrations in LIBERO, then fine-tunes SmolVLA on the resulting data. Built for the [Humanoid Robot Learning Intern challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221).

**Status:** Retargeting works: 16 reviewed episodes, 71,274 frames. SmolVLA has trained for 2,000 updates, but its first autonomous rollout failed to grasp the bowl.

**Next steps:** Resume training when a Colab GPU becomes available, give the initial approach more weight during sampling, and check the predictions before evaluating again.

## See it in action

| Human recording                                      | Retargeted robot demonstration                                         |
| ---------------------------------------------------- | ---------------------------------------------------------------------- |
| ![Human bowl transfer](docs/media/human_demo.gif)    | ![Successful retargeted bowl transfer](docs/media/retargeted_demo.gif) |
| [Watch video](docs/media/human_demo.mp4) · real time | [Watch video](docs/media/retargeted_demo.mp4) · **16× speed**          |

The robot above follows the retargeted human path using a programmed controller. Here is what the **trained VLA** does after 2,000 updates:

[![VLA evaluation: drift without grasp](docs/media/vla_step_002000.gif)](docs/media/vla_step_002000.mp4)

[Watch evaluation](docs/media/vla_step_002000.mp4) · **16× speed**. The wrist rotates and drifts without making contact with the bowl. Learning the initial approach is the current stumbling block.

## From phone video to robot actions

The task is to pick up the central black bowl and place it on the plate, using a mounted Franka Panda in LIBERO's `libero_spatial` suite. All demonstrations use the same simulation scene. See the [task configuration](configs/task.json).

The pipeline has four steps:

1. **Record and track.** An iPhone 13 mini films the transfer from a fixed overhead view. OpenCV tracks a marker on the hand relative to a printed ChArUco board.
2. **Annotate and retarget.** I mark grasp and release events manually. The transfer path is mapped onto the robot's grasp and destination, rotating and scaling its horizontal shape while preserving vertical deviations in metres.
3. **Replay and review.** A controller handles the approach, grasp, wrist orientation, placement and release. It follows the mapped path and checks contact and bowl drift. Successful episodes are reviewed before export.
4. **Train and evaluate.** SmolVLA learns from the rendered robot images, robot state and recorded actions. During evaluation, it controls the arm without help from the retargeting controller.

The human contribution is the transfer motion. Grasping and placement are engineered, and timing is adapted for the robot. The phone pixels are not used as VLA inputs; this experiment uses neither MediaPipe nor reinforcement learning.

## Setup

| Component        | Details                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Human recordings | 17 collected; one tracking failure excluded; 16 retained                                                                             |
| Printed markers  | ChArUco board: 5 × 7 squares, 30 mm squares, 22 mm markers. Hand marker: ID 49, 40 mm black square, `DICT_4X4_50`                    |
| Robot dataset    | 16 reviewed episodes, **71,274 frames at 20 Hz**                                                                                     |
| Observations     | External and wrist RGB images, 256 × 256; 8 state values: XYZ, rotation vector and two finger positions                              |
| Actions          | 7 normalized relative `OSC_POSE` commands: translation, rotation and gripper                                                         |
| Model            | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), about 99.9M trainable parameters                              |
| Training         | 2,000 updates; batch size 1; AdamW; learning rate `1e-5`; weight decay `0.01`; gradient clipping `1.0` (hyperparameters to be tuned) |
| Action chunks    | Predict 50 actions, execute 10 before predicting again                                                                               |
| Hardware         | MacBook Pro M2 for preparation, Colab T4 for training and evaluation                                                                 |

All 16 episodes are used for training. This is a proof of concept on one fixed scene, with no held-out split.

## What worked, and what still needs work

Printed markers were enough to recover useful human motion, and retargeting produced complete grasp–transfer–release episodes. Different recordings produced different transfer paths. The controller is slow, though, so the generated episodes and evaluations are long.

Training loss decreased, but that did not translate into a successful rollout:

| Measurement                           | Result                                                        |
| ------------------------------------- | ------------------------------------------------------------- |
| Mean loss, first → last 500 updates   | 1.063 → 0.580                                                 |
| Median loss, first → last 500 updates | 0.502 → 0.144                                                 |
| Autonomous evaluation                 | **0/1 successful**, recorded training scene, seed 42          |
| Rollout length                        | 6,000 steps: 300 simulated seconds, roughly 50 minutes to run |
| Finger–bowl contact                   | None                                                          |

Checks on recorded frames showed premature wrist rotation, a positive Y-command bias and a missed transition toward the bowl. The next training run will sample 30% of chunks from frames 0–350 and 70% from the remaining frames. This is prepared but has not been tested yet.

The 16 reviewed episodes are a curated dataset, not a measure of the retargeter's overall success rate. One policy rollout also isn't enough to judge reliability.

[Prediction plot](docs/results/startup_predictions.png) · [Retargeting report](docs/results/retarget_demo_001.json) · [Policy evaluation report](docs/results/vla_step_002000.json)

## How to run it

Simulation and training run in Linux/Colab. In a GPU runtime, install:

```bash
git clone https://github.com/huggingface/lerobot.git /content/lerobot
python -m pip install -e '/content/lerobot[libero,smolvla]'
python -m pip install num2words
```

Install before importing Transformers; restart the runtime if you installed `num2words` after importing it. Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` before importing the simulation libraries.

The reported runs used Python 3.13, LeRobot 0.6.2, hf-libero 0.1.4, robosuite 1.4.0 and MuJoCo 3.8.1.

### Replay the included human trajectory

[examples/demo_001](examples/demo_001) contains a tracked path, event annotations and camera calibration. From the repository root in the configured environment:

```bash
mkdir -p data/processed/example_demo
cp examples/demo_001/marker_smoothed.csv data/processed/example_demo/
cp examples/demo_001/human_events.json data/processed/example_demo/
printf 'n\n' | python scripts/retarget_episode.py \
  --project "$PWD" --demo example_demo \
  --output /content/example_demo_run
```

Use a new output folder for each run.

### Collect and train

| Notebook                                                                | Purpose                                                  |
| ----------------------------------------------------------------------- | -------------------------------------------------------- |
| [02 — Calibration](notebooks/02_camera_calibration.ipynb)               | Calibrate the phone camera                               |
| [08 — Retargeting](notebooks/08_batch_human_retargeting.ipynb)          | Track videos, annotate events, retarget and review       |
| [09 — Dataset export](notebooks/09_export_batch_training_dataset.ipynb) | Export approved episodes to LeRobot                      |
| [10 — SmolVLA](notebooks/10_train_smolvla_batch.ipynb)                  | Train, evaluate, inspect predictions and resume training |

Update `project` path and replace saved dataset/checkpoint IDs with your own. In notebook 10, run the section you need rather than running all sections together. Earlier notebooks contain the pilot experiments.

For local marker tracking, install `numpy` and `opencv-contrib-python` in a separate Python environment. Batch tools and checks:

```bash
python scripts/batch_retarget.py --help
python scripts/analyze_retarget_batch.py --project "$PWD"
python -m unittest discover -s tests -v
```

The repository includes code, notebooks, a small trajectory example, preview videos and selected results. Full recordings, training datasets and checkpoints are excluded from Git and are not published yet. You can replay the included example, but reproducing the full training run requires those additional files.

## A personal note

This started as a job application project, but it turned into something of a _Journey to the Centre of the Earth_. As I tried to get things working before the deadline, my curiosity kept taking me on detours: exploring alternative approaches, understanding how other people tackle these problems, and discovering what researchers and robotics companies would have done differently. I enjoyed learning about every stage, from recording human motion to training a VLA—including discovering how frustrating retargeting can be!

I used ChatGPT to help with design choices and implementation, questioning its suggestions and discussing alternatives as I went. I also read related papers and explored similar GitHub projects. I don't claim a novel method here. What surprised me most was how much I could learn with a phone, printed markers and a project built at home.

## References

- [Humanoid challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LeRobot](https://github.com/huggingface/lerobot)
- [SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla)
