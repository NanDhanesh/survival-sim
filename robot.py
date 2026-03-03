import numpy as np
from scipy import ndimage

# Size of the binary mask (8x8 grid of voxels)
MASK_DIM = 8

# Scale of the robot (edge length of a voxel)
# NOTE: this is very important as the simulator physics are configured to use this scale, more or less.
SCALE = 0.1

# Theoretical maximum masses and springs for a fully filled 8x8 grid
# masses: (MASK_DIM+1)^2 = 81 corner points
# springs: 72 horizontal + 72 vertical + 64 diag1 + 64 diag2 = 272
MAX_N_MASSES = (MASK_DIM + 1) ** 2  # 81
MAX_N_SPRINGS = 272


def load_robots(num_robots):
    return [sample_robot() for _ in range(num_robots)]


def sample_robot(p=0.55):
    """Randomly sample a robot with a connected voxel morphology."""
    mask = sample_mask(p)
    return robot_from_mask(mask)


def robot_from_mask(mask):
    """Convert a binary voxel mask to a robot dict with mass-spring geometry.

    The returned dict includes the mask itself (the genome) plus the
    derived masses and springs arrays.
    """
    masses, springs = mask_to_robot(mask)
    masses = masses * SCALE  # NOTE: scale of the robot geometry is KEY to stable simulation!
    return {
        "mask": mask.copy(),
        "n_masses": masses.shape[0],
        "n_springs": springs.shape[0],
        "masses": masses,
        "springs": springs,
    }


def mutate_mask(parent_mask, p_flip=0.05, min_voxels=3):
    """Mutate a binary voxel mask by flipping random cells.

    Ensures the result is a single connected component with at least
    *min_voxels* filled cells.  Falls back to the parent mask if the
    mutation produces an invalid organism.
    """
    child_mask = parent_mask.copy()
    flips = np.random.random(child_mask.shape) < p_flip
    child_mask = np.logical_xor(child_mask, flips).astype(int)

    # Extract largest connected component
    labeled, num_features = ndimage.label(child_mask)
    if num_features == 0:
        return parent_mask.copy()

    component_sizes = ndimage.sum(child_mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    child_mask = (labeled == largest_component).astype(int)

    # Reject if too small
    if child_mask.sum() < min_voxels:
        return parent_mask.copy()

    # Shift to bottom-left corner
    rows, cols = np.where(child_mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    component = child_mask[min_row : max_row + 1, min_col : max_col + 1]
    new_mask = np.zeros((MASK_DIM, MASK_DIM), dtype=int)
    component_height, component_width = component.shape
    new_mask[MASK_DIM - component_height : MASK_DIM, 0:component_width] = component
    return new_mask


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from original)
# ---------------------------------------------------------------------------

def voxel_to_masses(row, col):
    """Convert a voxel position to a list of mass coordinates.
    Each voxel has a mass at each of its four corners.
    """
    return [
        [row, col],
        [row, col + 1],
        [row + 1, col],
        [row + 1, col + 1],
    ]


def mask_to_robot(mask):
    """Convert a binary mask to a mass-spring robot geometry.

    Each voxel is represented by 4 masses and 6 springs.
    Masses are located at the corners of the voxel.
    Springs connect adjacent masses along the edges and diagonals.
    """
    spring_connections = [
        [0, 1],  # bottom left (bl) to bottom right (br)
        [0, 2],  # bl to top left (tl)
        [1, 3],  # br to top right (tr)
        [2, 3],  # tl to tr
        [0, 3],  # bl to tr
        [1, 2],  # br to tl
    ]
    masses = []
    springs = []
    rows, cols = np.where(mask)
    n_voxels = len(rows)
    for i in range(n_voxels):
        row = rows[i]
        col = cols[i]
        coords = voxel_to_masses(row, col)
        for c in coords:
            if c not in masses:
                masses.append(c)
        for a, b in spring_connections:
            ca = coords[a]
            cb = coords[b]
            ia = masses.index(ca)
            ib = masses.index(cb)
            s = [min(ia, ib), max(ia, ib)]
            if s not in springs:
                springs.append(s)
    masses = np.array(masses, dtype=np.float32)
    springs = np.array(springs, dtype=np.int32)
    return masses, springs


def sample_mask(p):
    """Sample a random binary mask, keep the largest connected component,
    and shift it to the bottom-left corner.
    """
    mask = np.random.uniform(0.0, 1.0, size=(MASK_DIM, MASK_DIM))
    mask = mask < p
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return sample_mask(p)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = labeled == largest_component
    rows, cols = np.where(mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    component = mask[min_row : max_row + 1, min_col : max_col + 1]
    new_mask = np.zeros((MASK_DIM, MASK_DIM), dtype=int)
    component_height, component_width = component.shape
    new_mask[MASK_DIM - component_height : MASK_DIM, 0:component_width] = component.astype(int)
    return new_mask
