"""Where each node of a growing tree sits in the side panel, keyed by its path from the root.

Nodes are keyed "", "L", "LR", … so a tree grown one split at a time never makes the layout jump.

A key is the path of turns from the root: "" is the root, "L" its left child, "LR" that child's
right child. Leaves are spread evenly left to right, each parent sits midway between its two
children, and a node's height drops with its depth.
"""

DIAGRAM_BOX = (0.06, 0.1, 0.88, 0.66)  # x0, y0, width, height inside the panel


def layout(leaf_keys, split_keys, box: tuple[float, float, float, float] = DIAGRAM_BOX
           ) -> dict[str, tuple[float, float]]:
    """Positions for every node of a tree, keyed by path.

    Leaves spread evenly left to right; parents centred over children; depth sets the height. Sorting the
    paths with "L" as 0, "R" as 2 and a 1 appended puts leaves of any depth in left-to-right order.

    Args:
        leaf_keys: The paths of the leaves.
        split_keys: The paths of the split nodes; each one's two children must be among the keys.
        box: Where the tree goes, as ``(x0, y0, width, height)`` inside the panel: the root on the top
            edge, the deepest leaves on the bottom one.

    Returns:
        ``(x, y)`` for every leaf and split, keyed by path.

    Example:
        >>> layout(["L", "R"], [""], box=(0.0, 0.0, 1.0, 1.0))
        {'L': (0.25, 0.0), 'R': (0.75, 0.0), '': (0.5, 1.0)}
    """
    x0, y0, width, height = box
    leaves = sorted(leaf_keys, key=lambda k: k.replace("L", "0").replace("R", "2") + "1")
    depth = max(len(key) for key in leaves) or 1
    x_of = {key: x0 + width * (i + 0.5) / len(leaves) for i, key in enumerate(leaves)}
    for key in sorted(split_keys, key=len, reverse=True):
        x_of[key] = (x_of[key + "L"] + x_of[key + "R"]) / 2
    return {key: (x, y0 + height * (1 - len(key) / depth)) for key, x in x_of.items()}
