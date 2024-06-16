import spatialdata
from spatialdata.transformations import Affine, set_transformation, \
    get_transformation


from spacem_lib.utils.spatialdata_utils import (
    COORD_SYS_GLOBAL,
    CYX,
    MICROMETER,
    YX,
    X,
    Y,
    initialize_spatialdata,
    sd_add_image,
)

# def write_transformation_to_spatialdata(
#         transformation: TransformMatrix2d, path: Path, modality: str,
#         identifier: Mapping[str, str]
# ):
path = "/scratch/bailoni/datasets/australia_project/data_2024/astrocytes_march_24/spacem_v2_andreas/spatialdata.zarr"
sdata = spatialdata.read_zarr(path)

for well_id in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:

    identifier = {"slide_id": "slide6",
                "well_id": well_id}
    from spacem_mosaic.tasks import IMAGE_NAME_TEMPLATE
    image_name_post = IMAGE_NAME_TEMPLATE.format(**identifier, modality="post_maldi")
    transf = get_transformation(
        sdata.images[image_name_post],
        # to_coordinate_system=YX,
    )

    matrix = transf.matrix
    matrix[1,1] *= 1.978125
    matrix[2,2] *= 1.978125
    transf.matrix = matrix


    image_name_pre = IMAGE_NAME_TEMPLATE.format(**identifier, modality="pre_maldi")
    set_transformation(
        sdata.images[image_name_pre],
        transformation=transf,
        write_to_sdata=sdata,
    )

