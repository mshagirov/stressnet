'''Plotting helpers shared by the training drivers.

Everything here writes static PNG artefacts with the Agg backend (matplotlib is
imported at module load and never displayed), except :func:`plot_scatter`,
which writes the interactive plotly HTML used by ``main.py``.
'''

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_scatter(y_tgt, y_pred, x_names, phase, save_dir):
    '''Interactive predicted-vs-true plotly scatter for the regression model.

    Axes span the Yv1-normalised range [-1.3, 1.3]; writes
    ``<save_dir>/<phase>_scatter.html``.
    '''
    plot_title = {'train': 'Training Data', 'val': 'Validation/Test data'}[phase]

    print(f"Plot: y_tgt {y_tgt.shape}; y_pred {y_pred.shape}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[-1.3, 1.3], y=[-1.3, 1.3],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5,
        showlegend=False,
    ))

    hover_texts = [
        f'{n}<br>(tgt={t:.3f}, pred={p:.3f})'
        for n, t, p in zip(x_names, y_tgt, y_pred, strict=True)
    ]
    fig.add_trace(go.Scatter(
        x=y_tgt, y=y_pred,
        mode='markers',
        text=hover_texts,
        hoverinfo='text',
        marker=dict(size=6),
        name=phase,
    ))

    fig.update_xaxes(range=[-1.3, 1.3], title='y_true [a.u.]')
    fig.update_yaxes(range=[-1.3, 1.3], title='y_predicted [a.u.]',
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(title=f'{plot_title}', width=600, height=600)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f'{phase}_scatter.html'
    fig.write_html(path)
    print(f'Saved: {path}')


def plot_history(losses: dict, accs: dict, save_path: Path, stage_boundary: int | None = None):
    '''Two-panel (loss, accuracy) per-epoch history; marks the unfreeze epoch.'''
    fig, axs = plt.subplots(1, 2, figsize=[12, 5])
    for ax, history, title in zip(axs, (losses, accs), ('Loss', 'Accuracy'), strict=True):
        for phase in history:
            ax.plot(history[phase], label=phase)
        if stage_boundary:
            ax.axvline(stage_boundary - 0.5, ls=':', c='k', alpha=.5,
                       label='unfreeze')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_classifier_predictions(preds_by_phase: dict, save_path: Path, class_names):
    '''One panel per phase: predicted probability vs true class, coloured by correctness.

    `preds_by_phase` maps a phase name to a :func:`stressnet.metrics.predict_classifier`
    frame and `class_names` lists the class labels in model output order (the
    ``p_<name>`` columns must be present). With two classes the y axis is
    P(second class) -- for the age task that is P(adult), exactly as before --
    otherwise it is the probability of the predicted class; the chance line is
    drawn at 1/n_classes.
    '''
    class_names = list(class_names)
    n_classes = len(class_names)
    p_cols = [f'p_{name}' for name in class_names]
    if n_classes == 2:
        ylabel = f'P({class_names[1]})'
    else:
        ylabel = 'P(predicted class)'

    fig, axs = plt.subplots(1, len(preds_by_phase),
                            figsize=[7*len(preds_by_phase), 6], squeeze=False)
    rng = np.random.default_rng(42)
    for ax, (phase, df) in zip(axs.ravel(), preds_by_phase.items(), strict=True):
        jitter = rng.uniform(-0.04, 0.04, size=len(df))
        colors = np.where(df['correct'].astype(bool), 'tab:green', 'tab:red')
        if n_classes == 2:
            y = df[p_cols[1]].to_numpy()
        else:
            y = df[p_cols].to_numpy().max(axis=1)
        ax.scatter(df['y_true'] + jitter, y, c=colors, alpha=.5, s=18)
        ax.axhline(1/n_classes, ls='--', c='r', alpha=.4)
        acc = df['correct'].mean()
        ax.set_title(f'{phase} (acc={acc:.3f})')
        ax.set_xlim(-0.25, n_classes - 1 + 0.25)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=(30 if n_classes > 2 else 0))
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_regression_predictions(preds_by_phase: dict, save_path: Path):
    '''One panel per phase: predicted vs true stiffness scatter (Yv1 units).

    `preds_by_phase` maps a phase name to the dict returned by
    :func:`stressnet.metrics.predict_regression`; a dashed 1-to-1 line and the
    [-1.25, 1.25] axis limits are kept from the original regression driver.
    '''
    fig, axs = plt.subplots(1, len(preds_by_phase), figsize=[7*len(preds_by_phase), 6],
                            squeeze=False)
    for ax, (phase, res) in zip(axs.ravel(), preds_by_phase.items(), strict=True):
        ax.scatter(res['y_tgt'], res['y_pred'], label=phase, alpha=.5)
        ax.plot([-1.25, 1.25], [-1.25, 1.25], '--', c='r', alpha=.25)
        ax.axis([-1.25, 1.25, -1.25, 1.25])
        ax.set_title(phase)
        ax.set_ylabel('$log(E_{pred})$ [a.u.]')
        ax.set_xlabel('$log(E_{tgt})$ [a.u.]')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')
