from pathlib import Path

def ensure_dir(path):
    """
    Ensure the directory exists.

    Parameters
    ----------
    path : str
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_plot(fig, save_path):
    """
    Save figure object into a directory
    
    Parameters
    ----------
    fig : figure object
    save_path : path to the directory
    """
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")