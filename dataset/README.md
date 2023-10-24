# GAPartNet Dataset

## Data Format

The GAPartNet dataset is built based on two exsiting datasets, PartNet-Mobility and AKB-48, from which the 3D object shapes are collected, cleaned, and equipped with new uniform GAPart-based semantics and poses annotations. The model_ids we use are provided in `render_tools/meta/{partnet_all_id_list.txt, akb48_all_id_list.txt}`.

### PartNet-Mobility

Four additional files accompany each object shape from PartNet-Mobility, providing annotations in the following formats:

- `semantics_gapartnet.txt`: This file contains link semantics. Each line corresponds to a link in the kinematic chain, as indicated in `mobility_annotation_gapartnet.urdf`, formatted as "[link_name] [joint_type] [semantics]".
- `mobility_annotation_gapartnet.urdf`: This document describes the kinematic chain, including our newly re-merged links and modified meshes. Each GAPart in the object shape corresponds to an individual link. We recommend using this file for annotation (semantics, poses) rendering and part properties queries.
- `mobility_texture_gapartnet.urdf`: This file also describes the kinematic chain but uses the original meshes. Each GAPart in the kinematic chain is not guaranteed to be an individual link. In our paper, we mentioned that since the GAPart semantics are newly defined, the meshes and annotations in the original assets may be inconsistent with our definition, which requires a finer level of detail. For example, in the original mesh for "Oven" or "Dishwasher," a line_fixed_handle and a hinge_door could be attached into a single .obj mesh file. To address this issue, we modified the meshes to separate the GAParts. However, these mesh modifications may have caused issues in the broken texture, resulting in poor quality in rendering. As a temporary solution, we provide this file and use the original meshes for texture rendering. The examplar code for the joint correspondence between the kinematic chains in `mobility_annotation_gapartnet.urdf` and `mobility_texture_gapartnet.urdf` can be found in our rendering toolkit.
- `link_annotation_gapartnet.json`: The json file contains GAPart semantics and pose of each link in the kinematic chain in `mobility_annotation_gapartnet.urdf`. Spefically, for each link, "link_name", "is_gapart", "category", "bbox" are provided, where "bbox" are the 3D bounding box position of the part in the rest state, i.e., all joint states (poses) are set to zero. The order of the eight vertices is as follows: [(-x,+y,+z), (+x,+y,+z), (+x,-y,+z), (-x,-y,+z), (-x,+y,-z), (+x,+y,-z), (+x,-y,-z), (-x,-y,-z)]. The coordinates of the vertices are in the world space.

### AKB-48

Three additional files accompany each object shape from AKB-48, providing annotations in the following formats:

- `semantics_gapartnet.txt`: This file contains link semantics. The format is the same as the file in PartNet-Mobility above.
- `mobility_annotation_gapartnet.urdf`: This document describes the kinematic chain. Each GAPart in the object shape corresponds to an individual link.
- `link_annotation_gapartnet.json`: The json file contains GAPart semantics and pose of each link in the kinematic chain in `mobility_annotation_gapartnet.urdf`. The format is the same as the file in PartNet-Mobility above.

## Data Split

The data splits used in our paper can be found in `render_tools/meta/{partnet_all_split.json, akb48_all_split.json}`. We split all 27 object categories into 17 seen and 10 unseen categories. Each seen category was further split into seen and unseen instances. This two-level split ensures that all GAPart classes exist in both seen and unseen object categories, which helps evaluate intra- and inter-category generalizability.

## Rendering Toolkit

We provide an example toolkit for rendering and visualizing our GAPartNet dataset, located in `render_tools/`. This toolkit relies on [SAPIEN](https://github.com/haosulab/SAPIEN). To use it, please check the requirements in `render_tools/requirements.txt` and install the required packages.

To render a single view of an object shape, use the `render_tools/render.py` script with the following command:

```shell
python render.py --dataset {DATASET} \
                 --model_id {MODEL_ID} \
                 --camera_idx {CAMERA_INDEX} \
                 --render_idx {RENDER_INDEX} \
                 --height {HEIGHT} \
                 --width {WIDTH} \
                 --ray_tracing {USE_RAY_TRACING} \
                 --replace_texture {REPLACE_TEXTURE}
```

The parameters are as follows:

- `DATASET`: The dataset to render. Use 'partnet' for PartNet-Mobility and 'akb48' for AKB-48.
- `MODEL_ID`: The ID of the object shape you want to render.
- `CAMERA_INDEX`: The index of the selected camera position range. This index is pre-defined in `render_tools/config_utils.py`.
- `RENDER_INDEX`: The index of the specific rendered view.
- `HEIGHT`: The height of the rendered image.
- `WIDTH`: The width of the rendered image.
- `USE_RAY_TRACING`: A boolean value specifying whether to use ray tracing for rendering. Use 'true' to enable and 'false' to disable.
- `REPLACE_TEXTURE`: A boolean value that determines whether to use the original texture or the modified texture for rendering. Set it to 'true' to use the original texture (better) and 'false' to use the modified. Note that this parameter is only valid for PartNet-Mobility, since AKB-48 does not need texture replacement.

To render the entire dataset, utilize the `render_tools/render_all_partnet.py` script and the `render_tools/render_all_akb48.py` with the following command:

``````shell
python render_all_partnet.py --ray_tracing {USE_RAY_TRACING} \
                             --replace_texture {REPLACE_TEXTURE} \
                             --start_idx {START_INDEX} \
                             --num_render {NUM_RENDER} \
                             --log_dir {LOG_DIR}

python render_all_akb48.py --ray_tracing {USE_RAY_TRACING} \
                           --start_idx {START_INDEX} \
                           --num_render {NUM_RENDER} \
                           --log_dir {LOG_DIR}
``````

The parameters are defined as follows:

- `USE_RAY_TRACING` and `REPLACE_TEXTURE`: These parameters are identical to those described earlier.
- `START_INDEX`: Specifies the starting render index, which is the same as the `RENDER_INDEX` mentioned previously.
- `NUM_RENDER`: Specifies the number of views to render for each object shape and camera range.
- `LOG_DIR`: The directory where the log files will be saved.

To visualize the rendering results, use the `render_tools/visualize.py` script with this command:

```shell
python visualize.py --model_id {MODEL_ID} \
                    --category {CATEGORY} \
                    --camera_position_index {CAMERA_INDEX} \
                    --render_index {RENDER_INDEX}
```

The parameters are as follows:

- `MODEL_ID`: The ID of the object shape to visualize.
- `CATEGORY`: The category of the object.
- `CAMERA_INDEX`: The index of the selected range for the camera position, pre-defined in `render_tools/config_utils.py`.
- `RENDER_INDEX`: The index of the view that you wish to visualize.


## Pre-processing Toolkit

In addition to the rendering toolkit, we also provide a pre-processing toolkit to convert the rendered results into our model's input data format. This toolkit loads the rendered results, generates a partial point cloud via back-projection, and uses Farthest-Point-Sampling (FPS) to sample points from the dense point cloud.

To use the toolkit, first install the PointNet++ library in `process_tools/utils/pointnet_lib` with the following command: `python setup.py install`. This installation will enable FPS performance on GPU. The library is sourced from [HalfSummer11/CAPTRA](https://github.com/HalfSummer11/CAPTRA), which is based on [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) and [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

To pre-process the rendered results, use the `process_tools/convert_rendered_into_input.py` script with the following command:

```shell
python convert_rendered_into_input.py --dataset {DATASET} \
                                      --data_path {DATA_PATH} \
                                      --save_path {SAVE_PATH} \
                                      --num_points {NUM_POINTS} \
                                      --visualize {VISUALIZE}
```

The parameters are as follows:

- `DATASET`: The dataset to pre-process. Use 'partnet' for PartNet-Mobility and 'akb48' for AKB-48.
- `DATA_PATH`: Path to the directory containing the rendered results.
- `SAVE_PATH`: Path to the directory where the pre-processed results will be stored.
- `NUM_POINTS`: The number of points to sample from the partial point cloud.
- `VISUALIZE`: A boolean value indicating whether to visualize the pre-processed results. Use 'true' to enable and 'false' to disable.

