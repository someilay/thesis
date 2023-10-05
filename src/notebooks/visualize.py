import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.lines as plt_lines

from typing import Any, Iterable
from collections import deque

DEFAULT_TITLE = 'animation'
DEFAULT_PATH = './'
DEFAULT_FIGSIZE = (4, 4)
DEFAULT_SUBPLOT_KWARGS = {
    'autoscale_on': False,
    'xlim': (-5, 5),
    'ylim': (-5, 5),
}
DEFAULT_AXES_GRID_KWARGS = {
    'color': 'black',
    'linestyle': '--',
    'linewidth': 0.8,
    'alpha': 0.5
}
DEFAULT_LINE_KWARGS = {'c': 'black', 'lw': 2}
DEFAULT_CIRCLE_KWARGS = {'c': 'blue', 'markersize': 6, 'zorder': 5}
DEFAULT_TRACE_KWARGS = {'c': 'blue', 'lw': 1, 'alpha': 0.3}
DEFAULT_CORNER_TEXT_KWARGS = {'x': 0.05, 'y': 0.8, 's': '', 'zorder': 10}


def animate(idx: int,
            points: list[np.ndarray],
            histories_x: list[deque],
            histories_y: list[deque],
            lines: dict[tuple[int, int], plt_lines.Line2D],
            circles: list[plt_lines.Line2D],
            traces: list[plt_lines.Line2D],
            corner_text: plt.Text,
            stats: list[tuple[str, list | np.ndarray]],
            trace_chain_len: float) -> Iterable[plt_lines.Artist]:
    these_x = [
        points[j][idx][0] for j in range(len(points))
    ]
    these_y = [
        points[j][idx][1] for j in range(len(points))
    ]

    if idx == 0:
        for h in histories_x:
            h.clear()
        for h in histories_y:
            h.clear()

    for i, (h_x, h_y) in enumerate(zip(histories_x, histories_y)):
        if not h_x and not h_y:
            h_x.appendleft(points[i][idx][0])
            h_y.appendleft(points[i][idx][1])
            continue
        if (points[i][idx][0] - h_x[0])**2 + (points[i][idx][1] - h_y[0])**2 > trace_chain_len ** 2:
            h_x.appendleft(points[i][idx][0])
            h_y.appendleft(points[i][idx][1])

    for (i, j) in lines.keys():
        lines[(i, j)].set_data([these_x[i], these_x[j]], [these_y[i], these_y[j]])

    for i, c in enumerate(circles):
        c.set_data([points[i][idx][0]], [points[i][idx][1]])
    for i, t in enumerate(traces):
        t.set_data(histories_x[i], histories_y[i])

    stat_text = ''
    for stat in stats:
        stat_text += rf'{stat[0]}:  {stat[1][idx]:.3f}'
        stat_text += '\n'
    if stat_text:
        corner_text.set_text(stat_text)

    return *circles, *lines.values(), *traces, corner_text


def visualize_n_points(
        points: tuple[np.ndarray, ...] | list[np.ndarray] | np.ndarray,
        lines_defs: dict[tuple[int, int], dict[str, Any] | None] | list[tuple[int, int]] | None = None,
        circles_kwargs: dict[int, dict[str, Any]] | list[dict[str, Any] | None] | None = None,
        traces_kwargs: dict[int, dict[str, Any]] | list[dict[str, Any] | None] | None = None,
        title: str | None = None,
        save_path: str | None = None,
        figsize: tuple[int, int] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        axes_grid_kwargs: dict[str, Any] | None = None,
        corner_text_kwargs: dict[str, Any] | None = None,
        show_axes: bool = False,
        stats: list[tuple[str, list | np.ndarray]] | None = None,
        dt: float = 0.01,
        trace_chain_len: float = 0.1,
        trace_chains_amount: int = 50,
        show: bool = True
) -> anim.FuncAnimation | None:
    # Preprocess arguments
    if isinstance(points, tuple):
        points = list(points)
    if isinstance(points, np.ndarray):
        points = list(points)
    if isinstance(lines_defs, list):
        lines_defs = {val: DEFAULT_LINE_KWARGS for val in lines_defs}
    if isinstance(circles_kwargs, dict):
        new_circles_kwargs = [DEFAULT_CIRCLE_KWARGS for _ in points]
        for i in sorted(circles_kwargs.values()):
            new_circles_kwargs[i] = circles_kwargs[i]
        circles_kwargs = new_circles_kwargs
    if isinstance(traces_kwargs, dict):
        new_traces_kwargs = [DEFAULT_TRACE_KWARGS for _ in points]
        for i in sorted(traces_kwargs.values()):
            new_traces_kwargs[i] = traces_kwargs[i]
        traces_kwargs = new_traces_kwargs
    if not lines_defs:
        lines_defs = dict()
    if not circles_kwargs:
        circles_kwargs = [DEFAULT_CIRCLE_KWARGS for _ in points]
    if not traces_kwargs:
        traces_kwargs = [DEFAULT_TRACE_KWARGS for _ in points]
    if not title:
        title = DEFAULT_TITLE
    if not save_path:
        save_path = DEFAULT_PATH
    if not figsize:
        figsize = DEFAULT_FIGSIZE
    if not subplot_kwargs:
        subplot_kwargs = DEFAULT_SUBPLOT_KWARGS
    if not axes_grid_kwargs:
        axes_grid_kwargs = DEFAULT_AXES_GRID_KWARGS
    if not corner_text_kwargs:
        corner_text_kwargs = DEFAULT_CORNER_TEXT_KWARGS
    if not stats:
        stats = []
    if not points:
        return None

    # Validate points history
    shape = points[0].shape
    length = len(points[0])
    for i, point in enumerate(points):
        if shape != point.shape:
            raise ValueError(
                f'Not all points history have the same shape. '
                f'points[0].shape = {shape} != {point.shape} = point[{i}]'
            )
    if len(shape) != 2:
        raise ValueError(f'To many dimensions for point history. Required 2, found {len(shape)}')
    if shape[1] != 2:
        raise ValueError(f'To many dimensions for point. Required 2, found {shape[1]}')

    fig: plt.Figure = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_subplot(**subplot_kwargs)
    ax.set_aspect('equal', adjustable='box')

    if show_axes:
        ax.grid(**axes_grid_kwargs)
    else:
        plt.axis('off')

    lines: dict[tuple[int, int], plt_lines.Line2D] = {
        adj: ax.plot([], [], '-', **line_kwargs)[0] for adj, line_kwargs in lines_defs.items()
    }
    circles: list[plt_lines.Line2D] = [
        ax.plot([], [], 'o', **circle_kwargs)[0] for circle_kwargs in circles_kwargs
    ]
    traces: list[plt_lines.Line2D] = [
        ax.plot([], [], '-', **trace_kwargs)[0] for trace_kwargs in traces_kwargs
    ]

    corner_text: plt.Text = ax.text(**corner_text_kwargs, transform=ax.transAxes)
    histories_x = [deque(maxlen=trace_chains_amount) for _ in range(len(points))]
    histories_y = [deque(maxlen=trace_chains_amount) for _ in range(len(points))]

    ani = anim.FuncAnimation(
        fig, animate, interval=dt, frames=length, blit=True,
        fargs=(points, histories_x, histories_y, lines, circles, traces, corner_text, stats, trace_chain_len)
    )

    save_at = os.path.join(save_path, title) + '.html'
    with open(save_at, 'w') as f:
        f.write(ani.to_jshtml())

    if show:
        plt.show()

    return ani
