from pydantic import BaseModel


class WorldInfo(BaseModel):
    world: dict
    osm_zoom_offset: int
    max_level: int


class ConfusionMatrixResponse(BaseModel):
    gt_labels: list[int]
    pred_labels: list[int]
    matrix: list[list[int]]
