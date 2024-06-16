import spatialdata
import anndata

from pathlib import Path
from spatialdata.models import TableModel

from spacem_lib.utils.spatialdata_utils import spatialdata_locked, \
    update_table, DEFAULT_TABLE_NAME, ensure_table_written


def read_adata_from_spatialdata_zarr(path: Path):
    sdata = spatialdata.read_zarr(path)
    return sdata.tables[DEFAULT_TABLE_NAME]

def write_adata_to_spatialdata_zarr(sdata_path: Path,
                                    adata: anndata.AnnData,
                                    table_name: str = DEFAULT_TABLE_NAME):
    sdata = spatialdata.read_zarr(sdata_path)

    if table_name in sdata.tables:
        del sdata.tables[table_name]

    table = TableModel.parse(
        adata,
        region=list(adata.obs["region"].astype(str).unique()),
        region_key="region",
        instance_key="instance_id",
    )
    sdata.tables[table_name] = table
    ensure_table_written(sdata, table_name=table_name)


root_dir = Path("/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24")

path_slide_5 = root_dir / "spacem_v2/spatialdata.zarr"
path_slide_6 = root_dir / "spacem_v2_6/spatialdata.zarr"
path_slide_4 = root_dir / "spacem_v2_4/spatialdata.zarr"
destination_path = root_dir / "backup/spacem_v2_6/spatialdata.zarr"

adata_5 = read_adata_from_spatialdata_zarr(path_slide_5)
adata_4 = read_adata_from_spatialdata_zarr(path_slide_4)
adata_6 = read_adata_from_spatialdata_zarr(path_slide_6)

adata_6 = anndata.concat([adata_4, adata_5, adata_6])

write_adata_to_spatialdata_zarr(destination_path, adata_6)
