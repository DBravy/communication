import os
import torch
import matplotlib.pyplot as plt
import math

import config
from model import ARCEncoder
from dataset import ARCDataset  # if you prefer a real ARC sample

# --------- helpers ---------
def tile_feature_map(tensor, max_channels=32, title="feature map"):
    """
    tensor: [C, H, W] on CPU
    Shows up to max_channels channels in a grid.
    """
    C, H, W = tensor.shape
    Cshow = min(C, max_channels)

    cols = int(math.ceil(math.sqrt(Cshow)))
    rows = int(math.ceil(Cshow / cols))

    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
    fig.suptitle(title)

    for i in range(Cshow):
        ax = fig.add_subplot(rows, cols, i + 1)
        fm = tensor[i]

        # normalize per-channel for visibility
        fm_min, fm_max = fm.min(), fm.max()
        if (fm_max - fm_min) > 1e-8:
            fm = (fm - fm_min) / (fm_max - fm_min)

        ax.imshow(fm.numpy(), interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"ch {i}", fontsize=8)

    plt.tight_layout()
    plt.show()


def load_pretrained_encoder(device):
    enc = ARCEncoder(
        num_colors=getattr(config, "NUM_COLORS", 10),
        embedding_dim=getattr(config, "EMBEDDING_DIM", 16),
        hidden_dim=getattr(config, "HIDDEN_DIM", 128),
        latent_dim=getattr(config, "LATENT_DIM", 512),
        num_conv_layers=getattr(config, "NUM_CONV_LAYERS", 3),
    ).to(device)

    ckpt_path = os.path.join(config.SAVE_DIR, "pretrained_encoder.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.eval()
    return enc


def get_one_arc_grid(device):
    """
    Returns a single ARC grid tensor [H, W] (dtype long) on the given device
    along with its actual size.
    Matches the constructor used in pretraining.
    """
    from dataset import ARCDataset  # uses (data_path, min_size=...) — no 'split'
    ds = ARCDataset(
        config.DATA_PATH, 
        min_size=getattr(config, "MIN_GRID_SIZE", 3),
        filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
        max_grids=getattr(config, 'MAX_GRIDS', None),
        num_distractors=0  # Feature map visualization doesn't need selection task
    )
    grid, size = ds[0]               # grid: [H, W] (long), size: (H, W)
    return grid.to(device), size


# --------- main ---------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_pretrained_encoder(device)

    # 1) Use a real ARC sample:
    grid, size = get_one_arc_grid(device)

    # 2) (Optional) Or construct a small toy grid manually
    # import torch
    # grid = torch.tensor([
    #     [1,1,0,0,0,2,2],
    #     [1,1,0,3,0,2,2],
    #     [0,0,0,3,0,0,0],
    #     [4,0,0,3,0,0,5],
    #     [4,4,0,0,0,5,5],
    # ], dtype=torch.long, device=device)
    # size = grid.shape  # (H, W)

    x = grid.unsqueeze(0)  # [1, H, W]
    feats = encoder.extract_feature_maps(x, sizes=[size])

    # Choose which maps to view:
    for name in ["conv1", "conv2", "dilated2", "dilated4", "combined", "refined", "pooled"]:
        fmap = feats[name].detach().cpu().squeeze(0)  # [C, H, W] (or [C, 4, 4] for pooled)
        tile_feature_map(fmap, max_channels=36, title=name)
