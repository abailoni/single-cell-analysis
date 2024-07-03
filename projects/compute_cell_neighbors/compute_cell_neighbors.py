import numpy as np
from typing import List
import pandas as pd
from pathlib import Path

from ome_zarr.types import PathLike
from napari_ome_zarr import napari_get_reader
import scanpy as sc

from nifty.graph import rag as nrag
from nifty.graph import UndirectedGraph
import vigra
from zarr.errors import ArrayNotFoundError


def load_ome_zarr_channels(ome_zarr_path: PathLike,
                           channels_to_select: List[str]):
    zarr_group_elements = napari_get_reader(ome_zarr_path)()

    # Loop over images in the zarr file:
    collected_images = {}
    full_channel_list = get_channel_list_in_ome_zarr(ome_zarr_path)

    for i, zarr_element in enumerate(zarr_group_elements):
        # Load metadata:
        metadata_dict = zarr_element[1]
        channel_axis = metadata_dict.get("channel_axis", None)
        channel_names = metadata_dict.get("name", None)
        channel_names = [channel_names] if isinstance(channel_names, str) else channel_names

        # Loop over channels in the zarr element:
        ch_image, channel_slice = None, None

        for ch_name in channels_to_select:
            assert ch_name in full_channel_list, f"Channel not found in ome-zarr file: {ch_name}. " \
                                                 f"Available channels are {full_channel_list}"

            if channel_names is not None:
                if ch_name not in channel_names:
                    continue
                # Load image:
                image = zarr_element[0][0].compute()
                # Find channel index:
                ch_idx = channel_names.index(ch_name)
                if channel_axis is not None:
                    image = image.take(ch_idx, channel_axis)
                else:
                    assert len(channel_names) == 1
                ch_image = np.squeeze(image)
            else:
                img_idx, ch_idx = ch_name.split("_ch_")
                if int(img_idx) == i:
                    # Load image:
                    assert channel_axis is not None, "Cannot deduce number of channels without channel axis info"
                    image = zarr_element[0][0].compute()
                    # Get channel:
                    ch_image = np.squeeze(image.take(ch_idx, channel_axis))

            if ch_image is not None:
                assert ch_image.ndim == 2, "Channels images should be 2D"
                assert ch_name not in collected_images, "Channel name found in multiple elements of the ome-zarr file"
                collected_images[ch_name] = ch_image

    return [collected_images[ch_name] for ch_name in channels_to_select]


def get_channel_list_in_ome_zarr(ome_zarr_path: PathLike):
    zarr_group_elements = napari_get_reader(ome_zarr_path)()

    # Loop over images in the zarr file:
    collected_channels = []

    for i, zarr_element in enumerate(zarr_group_elements):
        # Load metadata:
        metadata_dict = zarr_element[1]
        names = metadata_dict.get("name", None)
        channel_axis = metadata_dict.get("channel_axis", None)
        if names is not None:
            names = [names] if isinstance(names, str) else names
            collected_channels += names
        elif channel_axis is not None:
            image_shape = zarr_element[0][0].shape
            collected_channels += [f"{i}_ch_{ch}" for ch in range(image_shape[channel_axis])]

    return collected_channels


if __name__ == "__main__":
    # How many pixels to dilate the cells, before computing the neighbors:
    CELL_DILATION_RADIUS = 15

    project_dir = Path("/scratch/bailoni/projects/compute_cell_neighbors")
    root_pattern = "/scratch/abreu/{project}/D{donor}/{slide}/"
    metadata_csv_file_path = project_dir / "metadata_seadrugs_raw_FULL.csv"
    # project_dir = Path("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/bailoni/projects/compute_cell_neighbors")
    # root_pattern = "/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/abreu/{project}/D{donor}/{slide}/"
    # metadata_csv_file_path = project_dir / "metadata_seadrugs_raw_filtered.csv"

    target_pattern_anndata = project_dir / "results" / "{project}/D{donor}/{slide}/anndata/customdb_seadrugs_v2_neighbor_stats/{project}-D{donor}.{slide}.{row}{col}.cells.h5ad"


    # Load dataframe with dataset IDs:
    main_metadata = pd.read_csv(metadata_csv_file_path)
    if "cell_neighbors_processed" not in main_metadata.columns:
        main_metadata["cell_neighbors_processed"] = "False"
    else:
        main_metadata["cell_neighbors_processed"] = main_metadata["cell_neighbors_processed"].astype(str)



    project_dir.mkdir(parents=True, exist_ok=True)

    anndata_path_pattern = root_pattern + "anndata/customdb_seadrugs_v2/{project}-D{donor}.{slide}.{row}{col}.cells.h5ad"
    label_zarr_path_pattern = root_pattern + "microscopy.zarr/{row}{col}/pre_maldi"

    for index, row in main_metadata.iterrows():
        # Save metadata, so that we can resume the processing if something goes wrong:
        main_metadata.to_csv(metadata_csv_file_path, index=False)

        if row["cell_neighbors_processed"] == "True":
            # row["cell_neighbors_processed"] = False
            continue

        anndata_path = anndata_path_pattern.format(**row)
        label_zarr_path = label_zarr_path_pattern.format(**row)
        target_pattern_anndata_path = str(target_pattern_anndata).format(**row)

        # Take care of some inconsistencies in the anndata file names:
        if not Path(anndata_path).exists():
            # TODO: there is actually a column 'experiment` in the metadata that specifies how the AnnData naming is defined
            # Try to match any file with the following format:
            test_anndata_pattern = root_pattern + "anndata/customdb_seadrugs_v2/{project}*.{slide}.{row}{col}.cells.h5ad"
            test_anndata_path = test_anndata_pattern.format(**row)
            test_anndata_paths = list(Path(test_anndata_path).parent.glob(Path(test_anndata_path).name))
            if len(test_anndata_paths) == 1:
                anndata_path = test_anndata_paths[0]
                print(f"Found Anndata file: {anndata_path}")
            else:
                print(f"WARNING: Anndata path not found: {anndata_path}")
                main_metadata.at[index, "cell_neighbors_processed"] = "Anndata not found"
                continue

        # Load label ome-zarr:
        if not Path(label_zarr_path).exists():
            print(f"WARNING: Label zarr path not found: {label_zarr_path}")
            main_metadata.at[index, "cell_neighbors_processed"] = "Zarr file not found"
            continue

        segmentation_mask = load_ome_zarr_channels(label_zarr_path, ['cells'])[0]
        # try:
        # except (FileNotFoundError, ArrayNotFoundError):
        #     print(f"WARNING: Label zarr path not found or some problem loading it: {label_zarr_path}")
        #     main_metadata.at[index, "cell_neighbors_processed"] = "Faulty zarr segmentation: encountered some problem loading it"
        #     continue

        # Now load the Anndata object:
        adata = sc.read(anndata_path)

        # We check whether the segmentation matches the anndata file, by comparing cell areas:
        adata_cell_areas = adata.obs["area"]

        if adata.obs.index.astype('int').max() > segmentation_mask.max():
            print(f"WARNING: Number of cells {adata.obs.index.astype('int').max()} exceeds the number of cells in the segmentation mask. Original max: {segmentation_mask.max()} \n {anndata_path} \n {label_zarr_path}")
            main_metadata.at[index, "cell_neighbors_processed"] = "Number of cells exceeds segmentation mask (segmentation not matching)"
            continue

        segmentation_cell_areas = np.bincount(segmentation_mask.ravel())
        # In the adata object, only some cells are present (they were filtered) but the index IDs match the segmentation mask:
        # segmentation_cell_areas_filtered = segmentation_cell_areas
        segmentation_cell_areas_filtered = segmentation_cell_areas[adata.obs.index.astype('int')]
        # Check if the areas are similar:
        if not np.allclose(adata_cell_areas, segmentation_cell_areas_filtered, atol=1):
            print(f"WARNING: Cell areas in the segmentation mask do not match the Anndata object. {anndata_path} \n {label_zarr_path}")
            main_metadata.at[index, "cell_neighbors_processed"] = "Cell areas do not match segmentation mask"
            continue

        print(f"Processing {Path(anndata_path).name}...")

        # Experiment with growing the cells:
        if CELL_DILATION_RADIUS > 0:
            # Dilate the foreground mask by the specified radius:
            foreground_mask = segmentation_mask != 0
            foreground_mask_dilated = vigra.filters.multiBinaryDilation(
                foreground_mask.astype('uint8'),
                radius=CELL_DILATION_RADIUS
            )

            # Now compute some edge evidence between background and foreground,
            # using Distance Transform:
            dist_transform = vigra.filters.distanceTransform((segmentation_mask).astype('uint32'))

            # Now we grow the segmentation mask by using the Watershed algorithm:
            dilated_segm_mask = vigra.analysis.watershedsNew(
                dist_transform.astype('float32'),
                seeds=(segmentation_mask).astype('uint32'),
            )[0]

            # Apply again the dilated background mask to mask background:
            dilated_segm_mask = dilated_segm_mask * foreground_mask_dilated

            segmentation_mask = dilated_segm_mask

        # Construct Region Adjacency Graph from the segmentation masks:
        rag = nrag.gridRag(segmentation_mask.astype('uint32'))

        # Get edges of the graph:
        uv_Ids = rag.uvIds()
        number_of_cells = rag.numberOfNodes - 1

        # Find all uvIds that contain the background node:
        foreground_edges = np.logical_not((uv_Ids == 0).sum(axis=1) > 0)

        # Create a new graph with only the foreground edges/nodes:
        graph = UndirectedGraph(number_of_cells)
        graph.insertEdges(uv_Ids[foreground_edges] - 1)

        # Compute the number of neighbors for each cell:
        nb_neighbors = np.array(
            (
                [0] # Add back the background node to match the original cell indices
                + [
                    len(list(graph.nodeAdjacency(node)))
                    for node in range(graph.numberOfNodes)
                ]
            )
        )

        # Restrict nb_neighbors to the cells in the Anndata object:
        adata.obs["number_neighboring_cells"] = nb_neighbors[adata.obs.index.astype('int')]

        # Save the Anndata object:
        Path(target_pattern_anndata_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write(target_pattern_anndata_path)

        # Update metadata:
        main_metadata.at[index, "cell_neighbors_processed"] = "True"
        # Save metadata, so that we can resume the processing if something goes wrong:
    main_metadata.to_csv(metadata_csv_file_path, index=False)

