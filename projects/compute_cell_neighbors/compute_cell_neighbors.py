from napari_ome_zarr import napari_get_reader
import numpy as np
from ome_zarr.types import PathLike
from typing import Any, Callable, Dict, Iterator, List, Optional


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


import pandas as pd
from pathlib import Path
import scanpy as sc
from nifty.graph import rag as nrag
from nifty.graph import UndirectedGraph
import vigra

if __name__ == "__main__":
    # project_dir = Path("/scratch/bailoni/projects/compute_cell_neighbors")
    # root_pattern = "/scratch/abreu/{project}/D{donor}/{slide}/"

    CELL_DILATION_RADIUS = 8
    BRIGHTFIELD_CHANNEL = "Trans"

    project_dir = Path("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/bailoni/projects/compute_cell_neighbors")
    root_pattern = "/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/abreu/{project}/D{donor}/{slide}/"
    target_pattern_anndata = project_dir / "{project}/D{donor}/{slide}/anndata/customdb_seadrugs_v2_neighbor_stats/{project}-D{donor}.{slide}.{row}{col}.cells.h5ad"

    # Load dataframe with dataset IDs:
    main_metadata = pd.read_csv(project_dir / "metadata_seadrugs_raw.csv")
    if "neighbors_processed" not in main_metadata.columns:
        main_metadata["cell_neighbors_processed"] = False

    project_dir.mkdir(parents=True, exist_ok=True)

    anndata_path_pattern = root_pattern + "anndata/customdb_seadrugs_v2/{project}-D{donor}.{slide}.{row}{col}.cells.h5ad"
    label_zarr_path_pattern = root_pattern + "microscopy.zarr/{row}{col}/pre_maldi"

    for index, row in main_metadata.iterrows():
        if row["cell_neighbors_processed"]:
            continue

        anndata_path = anndata_path_pattern.format(**row)
        label_zarr_path = label_zarr_path_pattern.format(**row)
        target_pattern_anndata_path = str(target_pattern_anndata).format(**row)
        print(f"Processing {Path(anndata_path).name}...")

        # Load label ome-zarr:
        segmentation_mask = load_ome_zarr_channels(label_zarr_path, ['cells'])[0]

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

        # Now load the Anndata object:
        adata = sc.read(anndata_path)

        # Restrict nb_neighbors to the cells in the Anndata object:
        adata.obs["number_neighboring_cells"] = nb_neighbors[adata.obs.index.astype('int')]

        # Save the Anndata object:
        Path(target_pattern_anndata_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write(target_pattern_anndata_path)

        # Update metadata:
        main_metadata.at[index, "cell_neighbors_processed"] = True
        # Save metadata, so that we can resume the processing if something goes wrong:
        main_metadata.to_csv(project_dir / "metadata_seadrugs_raw.csv", index=False)

