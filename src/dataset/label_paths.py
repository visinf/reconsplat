"""Where a split's label shards live, resolved the same way for every dataset."""

from pathlib import Path


def resolve_label_dir(root: Path | str, split: str, subdir: str) -> Path:
    """Directory holding one kind of label shard for `split`, tolerating both layouts on disk.

    Two layouts exist for these datasets:

        <root>/<split>/<subdir>/*.npz    a fully-processed tree, where depth and cameras sit side by
                                         side under a single root (how DL3DV is processed)
        <root>/<split>/*.npz             a root holding one kind of label only (how RE10K is laid
                                         out, and how the camera-only bundles published for the test
                                         splits are shaped)

    Preferring the nested directory when it exists resolves both correctly, so the loaders no longer
    need dataset-specific rules. It also removes a trap: DL3DV used to select the nested layout from
    `labels_root == cameras_root and load_depth_labels`, which meant camera-only shards dropped into
    a processed root silently resolved to the flat path instead -- no error, just different cameras.
    Existence is the thing that actually matters, so that is what gets checked.
    """
    nested = Path(root) / split / subdir
    return nested if nested.is_dir() else Path(root) / split
