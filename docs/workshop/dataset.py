"""Single flat container for the fMRI dataset and spatial mappings."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import nibabel as nib
import numpy as np

@dataclass
class Dataset:
    """Responses, voxel/vertex labels, mappings, and grid geometry in one object.

    All voxel-indexed arrays use the same row order as ``voxel_mapping`` and
    every response array has shape (n_voxels, n_timepoints). Surface mappings
    store voxel *row indices*, not voxel coordinates.
    """
    name: str
    responses: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    voxel_mapping: np.ndarray
    voxel_hemisphere_id: np.ndarray
    voxel_roi_id: np.ndarray
    voxel_subcortical_id: np.ndarray
    roi_mapping: dict[int, str]
    subcortical_mapping: dict[int, str]
    mask: nib.spatialimages.SpatialImage
    metadata: dict[str, Any] = field(default_factory=dict)
    n_scans_used: dict[str, np.ndarray] = field(default_factory=dict)
    n_conditions_used: np.ndarray | None = None
    vertex_coordinates: np.ndarray | None = None
    vertex_hemisphere_id: np.ndarray | None = None
    vertex_roi_id: np.ndarray | None = None
    roi_subdivision_vertices: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    voxel_to_vertex: np.ndarray | None = None
    method1_vertex_to_voxel: np.ndarray | None = None
    method1_distance_mm: np.ndarray | None = None
    method2_vertex_voxel_offsets: np.ndarray | None = None
    method2_vertex_voxel_rows: np.ndarray | None = None

    def validate(self) -> None:
        n = len(self.voxel_mapping)
        if np.asarray(self.voxel_mapping).shape != (n, 3):
            raise ValueError("voxel_mapping must have shape (n_voxels, 3)")
        for name, arr in {
            "voxel_hemisphere_id": self.voxel_hemisphere_id,
            "voxel_roi_id": self.voxel_roi_id,
            "voxel_subcortical_id": self.voxel_subcortical_id,
        }.items():
            if np.asarray(arr).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},); got {np.shape(arr)}")
        if set(self.responses) != set(self.valid):
            raise ValueError("responses and valid must have identical keys")
        for name, response in self.responses.items():
            if np.asarray(response).ndim != 2 or response.shape[0] != n:
                raise ValueError(f"{name} response must have shape (n_voxels, n_timepoints)")
            validity = np.asarray(self.valid[name])
            if validity.shape != (n,):
                raise ValueError(f"valid[{name!r}] must have shape ({n},)")
        if self.vertex_coordinates is not None:
            nv = len(self.vertex_coordinates)
            for name, arr in {
                "vertex_hemisphere_id": self.vertex_hemisphere_id,
                "vertex_roi_id": self.vertex_roi_id,
                "method1_vertex_to_voxel": self.method1_vertex_to_voxel,
                "method1_distance_mm": self.method1_distance_mm,
            }.items():
                if arr is None or len(arr) != nv:
                    raise ValueError(f"{name} must have one value per surface vertex")
            for roi_name, subdivisions in self.roi_subdivision_vertices.items():
                for subdivision in ("low", "high"):
                    indices = np.asarray(subdivisions.get(subdivision, np.empty(0)), dtype=np.int64)
                    if np.any(indices < 0) or np.any(indices >= nv):
                        raise ValueError(
                            f"ROI subdivision {roi_name!r}/{subdivision} contains "
                            "out-of-range surface vertex indices"
                        )
        if self.method2_vertex_voxel_offsets is not None:
            offsets = self.method2_vertex_voxel_offsets
            if len(offsets) != len(self.vertex_coordinates) + 1:
                raise ValueError("Method-2 offsets must have n_vertices + 1 entries")
            if offsets[0] != 0 or offsets[-1] != len(self.method2_vertex_voxel_rows):
                raise ValueError("Method-2 CSR offsets do not match voxel-row indices")

    def __repr__(self) -> str:
        return (f"Dataset(name={self.name!r}, voxel_rows={len(self.voxel_mapping):,}, "
                f"responses={list(self.responses)}, "
                f"vertices={0 if self.vertex_coordinates is None else len(self.vertex_coordinates):,})")
