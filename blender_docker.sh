docker run --rm \
    -v $(pwd)/..:/$(pwd)/.. \
    -w $(pwd) \
    --name blender ikester/blender-autobuild:latest \
    -b $(pwd)/data/workspace_blender_example.blend \
    --python $(pwd)/src/preprocessing_furniture/blender_find_workspace.py
