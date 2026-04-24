# -*- coding: utf-8 -*-
"""
Style configuration for research visualizations.
Focused on premium academic aesthetics: Times New Roman, muted palettes, and clean layouts.
Unified font sizes for cross-figure consistency.
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.ticker import PercentFormatter


SINGLE_COLUMN_MM = 89
DOUBLE_COLUMN_MM = 180
SUPPLEMENT_WIDTH_MM = 170
MAP_WIDTH_MM = 180
FLOW_WIDTH_MM = 170
A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)

# Unified Font Sizes
FS_PANEL = 14.0
FS_TITLE = 11.0
FS_LABEL = 10.0
FS_TICK = 9.0
FS_ANNOT = 8.5
FS_LEGEND = 8.5
FS_LEGEND_TITLE = 9.0

MAIN_FIG_MARGINS = {
    'left': 0.075,
    'right': 0.985,
    'top': 0.955,
    'bottom': 0.11,
}

LAYOUT_PRESETS = {
    'main_landscape': {
        'width': 'double',
        'height_mm': 110,
        'margins': {'left': 0.075, 'right': 0.985, 'top': 0.955, 'bottom': 0.11},
        'gridspec': {'wspace': 0.28, 'hspace': 0.30},
    },
    'main_dense': {
        'width': 'double',
        'height_mm': 130,
        'margins': {'left': 0.08, 'right': 0.985, 'top': 0.955, 'bottom': 0.10},
        'gridspec': {'wspace': 0.28, 'hspace': 0.34},
    },
    'supplement_landscape': {
        'width': 'supplement',
        'height_mm': 94,
        'margins': {'left': 0.085, 'right': 0.985, 'top': 0.95, 'bottom': 0.14},
        'gridspec': {'wspace': 0.28, 'hspace': 0.28},
    },
    'supplement_dense': {
        'width': 'supplement',
        'height_mm': 112,
        'margins': {'left': 0.09, 'right': 0.985, 'top': 0.95, 'bottom': 0.12},
        'gridspec': {'wspace': 0.28, 'hspace': 0.30},
    },
    'flowchart': {
        'width': 'flow',
        'height_mm': 145,
        'margins': {'left': 0.05, 'right': 0.985, 'top': 0.965, 'bottom': 0.05},
        'gridspec': {'wspace': 0.0, 'hspace': 0.0},
    },
    'map': {
        'width': 'map',
        'height_mm': 138,
        'margins': {'left': 0.055, 'right': 0.985, 'top': 0.955, 'bottom': 0.07},
        'gridspec': {'wspace': 0.0, 'hspace': 0.12},
    },
}

SEQ_ORDER = [
    'S_to_H',
    'S_to_S',
    'Isolated_Storm',
    'Concurrent',
    'Isolated_Heatwave',
    'H_to_H',
    'H_to_S',
]

SEQ_LABELS = {
    'S_to_H': 'S->H',
    'H_to_H': 'H->H',
    'Concurrent': 'Concurrent',
    'Isolated_Storm': 'Isolated storm',
    'Isolated_Heatwave': 'Isolated heatwave',
    'S_to_S': 'S->S',
    'H_to_S': 'H->S',
}

# Premium Disturbance Colors
STORM_COLOR = '#2E86C1'      # Strong Blue
HEATWAVE_COLOR = '#E67E22'   # Burnt Orange
CONCURRENT_COLOR = '#8E44AD' # Royal Purple
RECOVERY_GREEN = '#2E7D32'   # Deep Green
FAILURE_RED = '#C62828'      # Deep Red

# Optimized Premium Palette
SEQ_COLORS = {
    'S_to_H': '#C0392B',            # Crimson (High Contrast)
    'S_to_S': STORM_COLOR,
    'Isolated_Storm': '#7F8C8D',    # Slate Grey (Baseline-like)
    'Concurrent': CONCURRENT_COLOR,
    'Isolated_Heatwave': HEATWAVE_COLOR,
    'H_to_H': '#D35400',            # Pumpkin Orange
    'H_to_S': '#3498DB',            # Sky Blue
}

NEUTRAL = '#7F8C8D'
GRID = '#F2F4F7'
AXIS_BLACK = '#2C3E50'
LIGHT_FILL = '#F8F9FA'


def _blend_with_white(color, amount=0.55):
    r, g, b = to_rgb(color)
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )


def mm_to_inch(mm):
    return mm / 25.4


def publication_size(width='double', height_mm=120):
    width_map = {
        'single': SINGLE_COLUMN_MM,
        'double': DOUBLE_COLUMN_MM,
        'supplement': SUPPLEMENT_WIDTH_MM,
        'map': MAP_WIDTH_MM,
        'flow': FLOW_WIDTH_MM,
    }
    width_mm = width_map.get(width, DOUBLE_COLUMN_MM)
    return (mm_to_inch(width_mm), mm_to_inch(height_mm))


def a4_size(orientation='landscape'):
    return A4_LANDSCAPE if orientation == 'landscape' else A4_PORTRAIT


def apply_pub_style():
    sns.set_theme(style='white', context='paper')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.facecolor': 'white',
        'savefig.bbox': None,
        'savefig.pad_inches': 0.01,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FS_TICK,
        'axes.titlesize': FS_TITLE,
        'axes.titleweight': 'bold',
        'axes.labelsize': FS_LABEL,
        'axes.labelweight': 'bold',
        'axes.linewidth': 0.9,
        'axes.edgecolor': AXIS_BLACK,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 4.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.pad': 3.0,
        'ytick.major.pad': 3.0,
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'grid.alpha': 1.0,
        'legend.fontsize': FS_LEGEND,
        'legend.title_fontsize': FS_LEGEND_TITLE,
        'legend.frameon': False,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def clean_axis(ax, grid_axis='none'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', direction='out', length=4.5, width=0.9, color=AXIS_BLACK)
    ax.tick_params(axis='both', which='minor', direction='out', length=2.5, width=0.7, color=AXIS_BLACK)

    ax.grid(False)
    if grid_axis == 'x':
        ax.xaxis.grid(True, linestyle=(0, (5, 6)), color=GRID, linewidth=0.7, zorder=0)
    elif grid_axis == 'y':
        ax.yaxis.grid(True, linestyle=(0, (5, 6)), color=GRID, linewidth=0.7, zorder=0)
    elif grid_axis == 'both':
        ax.xaxis.grid(True, linestyle=(0, (5, 6)), color=GRID, linewidth=0.65, zorder=0)
        ax.yaxis.grid(True, linestyle=(0, (5, 6)), color=GRID, linewidth=0.65, zorder=0)


def set_square_panel(ax, enabled=True):
    if enabled and hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1)


def add_panel_label(ax, label, x=-0.05, y=1.05):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=FS_PANEL,
        fontweight='bold',
        color=AXIS_BLACK,
    )


def set_percent_axis(ax, axis='y', xmax=1.0):
    formatter = PercentFormatter(xmax=xmax, decimals=0)
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def seq_color(category, muted=False):
    color = SEQ_COLORS.get(category, NEUTRAL)
    if muted and category != 'S_to_H':
        return _blend_with_white(color, amount=0.30)
    return color


def seq_fill(category, amount=0.6):
    return _blend_with_white(seq_color(category), amount=amount)


def ordered_sequences(categories, exclude=None):
    exclude_set = set(exclude or [])
    category_set = set(categories)
    return [seq for seq in SEQ_ORDER if seq in category_set and seq not in exclude_set]


def apply_main_figure_margins(fig, **overrides):
    margins = MAIN_FIG_MARGINS.copy()
    margins.update(overrides)
    fig.subplots_adjust(**margins)


def layout_preset(name):
    if name not in LAYOUT_PRESETS:
        raise KeyError(f'Unknown layout preset: {name}')
    preset = LAYOUT_PRESETS[name]
    return {
        'size': publication_size(preset['width'], preset['height_mm']),
        'margins': preset['margins'].copy(),
        'gridspec': preset['gridspec'].copy(),
    }


def create_figure(preset_name, override_height_mm=None):
    preset = layout_preset(preset_name)
    if override_height_mm is not None:
        preset['size'] = (preset['size'][0], mm_to_inch(override_height_mm))
    fig = plt.figure(figsize=preset['size'])
    return fig, preset


def apply_layout(fig, preset=None, **overrides):
    margins = MAIN_FIG_MARGINS.copy()
    if preset is not None:
        margins.update(layout_preset(preset)['margins'])
    margins.update(overrides)
    fig.subplots_adjust(**margins)


def make_gridspec(fig, nrows, ncols, preset=None, **kwargs):
    gridspec_kwargs = {}
    if preset is not None:
        gridspec_kwargs.update(layout_preset(preset)['gridspec'])
    gridspec_kwargs.update(kwargs)
    return fig.add_gridspec(nrows, ncols, **gridspec_kwargs)


def save_publication_figure(fig, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.02)
    svg_path = os.path.splitext(output_path)[0] + '.svg'
    fig.savefig(svg_path, format='svg', facecolor='white', bbox_inches='tight', pad_inches=0.02)
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    fig.savefig(pdf_path, format='pdf', facecolor='white', bbox_inches='tight', pad_inches=0.02)
