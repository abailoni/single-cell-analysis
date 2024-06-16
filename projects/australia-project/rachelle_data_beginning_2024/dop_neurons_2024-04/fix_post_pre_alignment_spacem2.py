import spatialdata
from spatialdata.transformations import Affine, set_transformation, \
    get_transformation

path = "/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24/spacem_v2/spatialdata.zarr"
sdata = spatialdata.read_zarr(path)

for slide_id in ["4",
                 "5", "6"
                 ]:
    for well_id in [
        "A1",
        "A2",
                    "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"
    ]:
        print(f"Processing slide {slide_id}, well {well_id}...")
        identifier = {"slide_id": f"slide{slide_id}",
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
        # matrix[1,1] *= 1
        # matrix[2,2] *= 1
        transf.matrix = matrix


        image_name_pre = IMAGE_NAME_TEMPLATE.format(**identifier, modality="pre_maldi")
        set_transformation(
            sdata.images[image_name_pre],
            transformation=transf,
            write_to_sdata=sdata,
        )

