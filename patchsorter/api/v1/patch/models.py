from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class PatchResponse(BaseModel):
    patch_id: int
    patch_uid: UUID | None = None
    label_class_id: int
    image_id: int
    downsample_factor: float
    centroid_x: float | None = None
    centroid_y: float | None = None
    polygon: str | None = None
    # Prediction columns (None when no prediction exists for this patch)
    embed_x: float | None = None
    embed_y: float | None = None
    grid_cell_i: int | None = None
    grid_cell_j: int | None = None
    pred_label_class_id: int | None = None
    event_ts: datetime | None = None
    priority: int | None = None


class LabelAssignByPolygonRequest(BaseModel):
    polygon: dict[str, Any]



class SampleByPointRequest(BaseModel):
    x: float
    y: float


class LabelAssignResponse(BaseModel):
    updated: int
