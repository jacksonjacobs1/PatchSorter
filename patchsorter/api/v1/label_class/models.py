from datetime import datetime

from pydantic import BaseModel


class LabelClassResponse(BaseModel):
    label_class_id: int
    project_id: int | None
    name: str
    color_code: str | None = None
    event_ts: datetime

    model_config = {"from_attributes": True}


class LabelClassCreate(BaseModel):
    name: str
    color_code: str | None = None


class LabelClassDefaultResponse(BaseModel):
    color_code: str


class LabelClassUpdate(BaseModel):
    name: str | None = None
    color_code: str | None = None
