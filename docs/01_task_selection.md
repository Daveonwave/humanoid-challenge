# Step 1: initial task selection

Decision date: 2026-10-08.

## Selected task

`libero_spatial / pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`

The upstream task definition identifies `akita_black_bowl_1` as the center bowl and `plate_1` as its destination. Its goal predicate is `(On akita_black_bowl_1 plate_1)`.

## Why start here?

The required behavior is a short reach, grasp, lift, transfer and release sequence. A small bowl and plate allow a practical real-world approximation. There is no drawer opening or articulated-object manipulation in the goal. This is a practical choice, not a measured claim that it is LIBERO's easiest task.

The scene also includes a second black bowl, cookies, a ramekin, a cabinet and a stove. The second bowl makes target selection meaningful. Inspect the rendered scene before deciding the physical layout and object dimensions. Do not remove distractors while presenting results as the original benchmark.

## Evaluation contract

Use the environment's success predicate as the primary benchmark metric. Separately record whether the gripper released the bowl and the placement remains stable; these are extra diagnostics, not changes to the official predicate. Freeze evaluation initial states before model tuning. Choose the episode horizon after inspecting the installed environment and replaying a demonstration.

One task does not demonstrate language understanding. Later work can compare instructions requiring different actions in the same scene.

## Next step: Colab simulator smoke test

1. Select a Linux GPU runtime and record Python, GPU and package versions.
2. Install compatible pinned LeRobot/LIBERO dependencies and configure headless rendering.
3. Enumerate tasks in `libero_spatial`; find the exact task by name and print its runtime index and language instruction.
4. Reset the environment and save front and wrist camera frames.
5. Inspect observation keys, shapes, action range and controller settings.
6. Step the environment briefly and save a video. Use verified gripper conventions; zero is not automatically a neutral gripper action.
7. Replay an existing task demonstration to check control and success detection.

Acceptance: a reproducible notebook showing the correct scene, observation/action metadata, and a saved video. Demonstration replay should validate task success before human-data retargeting begins.

## Sources inspected

- Task registry: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py
- Task definition: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl

These references were inspected on the upstream master branch, not pinned to a commit. Record the actual installed revision during setup.
