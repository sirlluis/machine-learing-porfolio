from pathlib import Path

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_plot(fig, save_path):
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")