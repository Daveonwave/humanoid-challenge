"""Compare current batch runs without loading image datasets or rerunning simulation."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def resample_path(path, times, points=101):
    path, times = np.asarray(path, float), np.asarray(times, float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) != len(times) or len(path) < 2:
        raise ValueError('Expected matching Nx3 path and N timestamps')
    if not np.isfinite(path).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Nonfinite path or non-increasing timestamps')
    u = (times-times[0])/(times[-1]-times[0])
    grid = np.linspace(0, 1, points)
    sampled = np.column_stack([np.interp(grid, u, path[:,i]) for i in range(3)])
    # Subtract the straight start-to-end connection to isolate curvature/lift.
    residual = sampled - ((1-grid[:,None])*sampled[0] + grid[:,None]*sampled[-1])
    return sampled, residual


def analyze(project):
    project = Path(project)
    manifest = json.loads((project/'configs/retarget_batch.json').read_text())
    output = project/'outputs/batch_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    rows, phase_rows, paths, residuals, skipped = [], [], {}, {}, []
    for demo, entry in manifest['demos'].items():
        if not entry.get('run'):
            skipped.append((demo, 'No attempted run')); continue
        run = project/entry['run']
        try:
            report = json.loads((run/'report.json').read_text())
            if report['status'] == 'running':
                skipped.append((demo, 'Run not finalized; skipped')); continue
            with np.load(run/'mapped_path.npz') as saved:
                path = saved['path'].copy()
                times = saved['human_times'].copy()
            sampled, residual = resample_path(path, times)
            with h5py.File(run/'episode.h5', 'r') as f:
                n = int(f.attrs['committed_steps'])
                fps = float(f.attrs['fps'])
                if fps <= 0 or len(f['phase']) < n:
                    raise ValueError('Invalid fps or incomplete phase data')
                phases = f['phase'].asstr()[:n]
                # Scalar arrays only: do not load expensive image datasets.
                state = f['state'][:n, :3]
                actions = f['action'][:n, :3]
            groups = np.array([re.sub(r'^human_transfer_\d+$', 'human_transfer', s) for s in phases])
            transfer = groups == 'human_transfer'
            transfer_steps = int(transfer.sum())
            for phase in dict.fromkeys(groups):
                count = int((groups == phase).sum())
                phase_rows.append(dict(demo=demo, phase=phase, steps=count,
                                       simulated_seconds=count/fps, percent_steps=100*count/max(n,1)))
            adjacent_transfer = transfer[1:] & transfer[:-1]
            displacements = np.linalg.norm(np.diff(state,axis=0),axis=1)
            speeds = displacements[adjacent_transfer]*fps*1000
            config = report.get('config',{})
            limit = config.get('translation_limit')
            at_limit = (np.any(np.abs(actions[transfer]) >= float(limit)*.99,axis=1).mean()
                        if limit and transfer_steps else np.nan)
            rows.append(dict(demo=demo, status=report['status'],
                training_eligible=report.get('training_eligible',False),run=str(run),
                total_steps=n,total_simulated_s=n/fps,
                human_transfer_s=float(times[-1]-times[0]),
                robot_transfer_s=transfer_steps/fps,
                transfer_duration_ratio=(transfer_steps/fps)/(times[-1]-times[0]),
                mapped_path_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()*1000),
                wrist_lift_above_start_mm=float((path[:,2].max()-path[0,2])*1000),
                maximum_residual_mm=float(np.linalg.norm(residual,axis=1).max()*1000),
                median_transfer_speed_mm_s=float(np.median(speeds)) if len(speeds) else np.nan,
                fraction_transfer_commands_at_translation_limit=float(at_limit)))
            paths[demo], residuals[demo] = sampled, residual
        except (OSError, KeyError, ValueError) as exc:
            skipped.append((demo, str(exc)))
    summary = pd.DataFrame(rows)
    timing = pd.DataFrame(phase_rows)
    summary.to_csv(output/'episode_summary.csv',index=False)
    timing.to_csv(output/'phase_timing.csv',index=False)
    (output/'skipped.json').write_text(json.dumps(skipped,indent=2))
    if summary.empty:
        print('No completed runs with mapped paths and recordings available.', skipped)
        return summary, timing, pd.DataFrame()
    names = list(paths)
    # RMS Euclidean separation at equal normalized human-transfer time, in mm.
    distance = pd.DataFrame([[float(np.sqrt(np.mean(np.sum((residuals[a]-residuals[b])**2,axis=1)))*1000)
                             for b in names] for a in names],index=names,columns=names)
    distance.to_csv(output/'pairwise_path_shape_rms_mm.csv')
    fig, axes = plt.subplots(1,3,figsize=(17,5))
    u = np.linspace(0,1,101)
    for name in names:
        path = paths[name]*1000
        axes[0].plot(path[:,0],path[:,1],label=name)
        axes[1].plot(u,path[:,2]-path[0,2],label=name)
    axes[0].set(xlabel='World X (mm)',ylabel='World Y (mm)',title='Mapped transfer paths (commanded)')
    axes[0].axis('equal')
    axes[1].set(xlabel='Normalized human transfer time',ylabel='Wrist height above start (mm)',title='Lift profiles')
    axes[1].legend(fontsize=7,ncol=2)
    im=axes[2].imshow(distance.to_numpy(),cmap='viridis',vmin=0)
    axes[2].set_xticks(range(len(names)),names,rotation=90)
    axes[2].set_yticks(range(len(names)),names)
    axes[2].set_title('Shape difference: RMS mm\nstraight endpoint connection removed')
    fig.colorbar(im,ax=axes[2])
    fig.tight_layout();fig.savefig(output/'path_variation.png',dpi=160);plt.show()
    pivot=timing.pivot(index='demo',columns='phase',values='simulated_seconds').fillna(0)
    ax=pivot.plot.bar(stacked=True,figsize=(13,6))
    ax.set(ylabel='Simulated seconds',title='Episode time by phase (not wall-clock runtime)')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    ax.figure.tight_layout();ax.figure.savefig(output/'phase_timing.png',dpi=160);plt.show()
    print('Saved diagnostics:', output)
    print('Comparisons include failed finalized runs; check status before interpreting partial transfer timings.')
    print('Shape differences use normalized human time, so they can reflect speed-profile differences as well as geometry.')
    print('Similar paths are not necessarily duplicate inputs. No universal pass/fail diversity threshold is applied.')
    if skipped: print('Skipped:',skipped)
    return summary,timing,distance


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args()
    summary,timing,distance=analyze(args.project)
    print(summary.to_string(index=False))
