from datetime import datetime

from pydantic import BaseModel


class ProjectResponse(BaseModel):
    project_id: int
    project_name: str
    description: str | None = None
    creation_ts: datetime | None = None
    total_patches: int | None = None


class ProjectStatsResponse(BaseModel):
    num_images: int
    patch_size: int | None
    num_label_classes: int
    total_objects: int
    labeled_count: int
    creation_date: datetime | None
    modification_date: datetime | None


class CreateProjectRequest(BaseModel):
    name: str
    description: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None
