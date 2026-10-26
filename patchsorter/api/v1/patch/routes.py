from fastapi import APIRouter, HTTPException, Query, Response
from shapely.geometry import shape
from sqlalchemy import text

from patchsorter.api.v1.confusion_matrix.utils import (
    _parse_label_pairs,
    _world_to_grid_bbox,
)
from patchsorter.api.v1.patch.models import (
    LabelAssignByPolygonRequest,
    LabelAssignResponse,
    PatchResponse,
)
from patchsorter.db.head_client import get_client as get_head_client
from patchsorter.db.head_client.models import build_table_name
from patchsorter.db.head_client.patch import PatchStore
from patchsorter.db.head_client.settings import SettingsStore

router = APIRouter()


@router.get("/projects/{project_id}/patches/", response_model=list[PatchResponse])
def list_patches(
    project_id: int,
    cursor: int = Query(
        default=0,
        description="Keyset cursor: last seen patch_id (exclusive lower bound)",
    ),
    limit: int = Query(default=20, ge=1, le=500),
    x_min: float | None = Query(default=None),
    y_min: float | None = Query(default=None),
    x_max: float | None = Query(default=None),
    y_max: float | None = Query(default=None),
    lp: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Label pair filter: repeat for each pair as 'gt,pred' (e.g. lp=0,1&lp=2,2)",
    ),
) -> list[PatchResponse]:
    label_pairs = _parse_label_pairs(lp)
    client = get_head_client()
    with client.get_session() as session:
        # Only enable this when testing EXPLAIN ANALYZE
        # session.execute(text("SET LOCAL citus.enable_local_execution TO OFF"))
        store = PatchStore(project_id, session)
        if (
            x_min is not None
            and y_min is not None
            and x_max is not None
            and y_max is not None
        ):
            settings_store = SettingsStore(session)
            max_level = settings_store.get_int("max_level", project_id)
            world_size = settings_store.get_int("world_size", project_id)
            try:
                i_min, j_min, i_max, j_max = _world_to_grid_bbox(
                    x_min, y_min, x_max, y_max, max_level, max_level, world_size
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            rows = store.get_patches_within_grid_bbox(
                i_min,
                i_max,
                j_min,
                j_max,
                cursor=cursor,
                limit=limit,
                include_image=False,
                label_pairs=label_pairs,
            )
        else:
            rows = store.fetch_predicted(
                cursor=cursor, limit=limit, include_image=False, label_pairs=label_pairs
            )
    return [PatchResponse(**r) for r in rows]


@router.get("/projects/{project_id}/patches/{patch_id}", response_model=PatchResponse)
def get_patch(project_id: int, patch_id: int) -> PatchResponse:
    client = get_head_client()
    with client.get_session() as session:
        store = PatchStore(project_id, session)
        rows = store._paginated_pred_join(
            pred_filter_sql="pu.patch_id = :patch_id_filter",
            pred_params={"patch_id_filter": patch_id},
            cursor=0,
            limit=1,
            include_image=False,
        )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Patch {patch_id} not found or has no prediction in project {project_id}",
        )
    return PatchResponse(**rows[0])


@router.get("/projects/{project_id}/patches/{patch_id}/image")
def get_patch_image(project_id: int, patch_id: int) -> Response:
    client = get_head_client()
    with client.get_session() as session:
        tbl = build_table_name(project_id)
        row = (
            session.execute(
                text(f"SELECT patch_image FROM {tbl} WHERE patch_id = :patch_id"),
                {"patch_id": patch_id},
            )
            .mappings()
            .first()
        )
    if not row or not row.get("patch_image"):
        raise HTTPException(status_code=404, detail="Patch image not found")
    return Response(
        content=row["patch_image"],
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=31536000",
        },
    )


@router.post("/projects/{project_id}/patches/", response_model=LabelAssignResponse)
def assign_labels_by_ids(
    project_id: int,
    patch_ids: list[int] = Query(..., description="Patch IDs to relabel"),  # noqa: B008
    label_class_id: int = Query(..., description="Ground-truth label class to assign"),
) -> LabelAssignResponse:
    client = get_head_client()
    with client.get_session() as session:
        store = PatchStore(project_id, session)
        updated = store.bulk_update_labels_by_ids(patch_ids, label_class_id)
    return LabelAssignResponse(updated=updated)


@router.post(
    "/projects/{project_id}/patches/polygonassign", response_model=LabelAssignResponse
)
def assign_labels_by_polygon(
    project_id: int,
    body: LabelAssignByPolygonRequest,
    label_class_id: int = Query(..., description="Ground-truth label class to assign"),
    lp: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Label pair filter: repeat for each pair as 'gt,pred' (e.g. lp=0,1&lp=2,2)",
    ),
) -> LabelAssignResponse:
    polygon = shape(body.polygon)
    label_pairs = _parse_label_pairs(lp)
    client = get_head_client()
    with client.get_session() as session:
        settings_store = SettingsStore(session)
        max_level = settings_store.get_int("max_level", project_id)
        world_size = settings_store.get_int("world_size", project_id)
        x_min, y_min, x_max, y_max = polygon.bounds
        i_min, j_min, i_max, j_max = _world_to_grid_bbox(
            x_min, y_min, x_max, y_max, max_level, max_level, world_size
        )
        store = PatchStore(project_id, session)
        updated = store.bulk_update_labels_by_polygon_bbox(
            i_min,
            i_max,
            j_min,
            j_max,
            polygon_wkt=polygon.wkt,
            label_class_id=label_class_id,
            label_pairs=label_pairs,
        )
    return LabelAssignResponse(updated=updated)
