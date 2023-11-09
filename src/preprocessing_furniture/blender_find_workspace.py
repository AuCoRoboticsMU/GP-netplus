import bpy
import bmesh
import os
import numpy as np

MESH_DATA_PATH = '/home/anna/Grasping/data/'

def find_z_faces(z_faces, face, z_values):
    """
    Recursively find all adjacent faces that are perpendicular to the Z-axis
    and add them to the z_faces list. abs(face.normal.angle(z_axis)) < 0.01 or
    """

    if abs(face.normal.angle(z_axis)) > (np.pi - 0.02) or abs(face.normal.angle(z_axis)) < 0.02:
        z_faces.append(face)
        for v in face.verts:
            z_values.append(v.co.z)


def get_adjacent_faces(face, all_faces):
    """
    Get a list of all faces adjacent to the given face.
    """
    adjacent_faces = []
    for edge in face.edges:
        for adjacent_face in edge.link_faces:
            if adjacent_face in all_faces and adjacent_face != face:
                adjacent_faces.append(adjacent_face)
    return adjacent_faces


def group_adjacent_faces(faces):
    """
    Group a set of faces into regions of adjacent faces.
    """
    regions = []
    for face in faces:
        # Check if this face is already in a region
        in_region = False
        for region in regions:
            if face in region:
                in_region = True
                break
        if in_region:
            continue

        # Find all adjacent faces to this face
        region_faces = [face]
        region_changed = True
        while region_changed:
            region_changed = False
            for region_face in region_faces:
                adjacent_faces = get_adjacent_faces(region_face, faces)
                for adjacent_face in adjacent_faces:
                    if adjacent_face not in region_faces:
                        region_faces.append(adjacent_face)
                        region_changed = True

        # Add this region to the list
        regions.append(region_faces)

    return regions


root = '{}/Shapenet_furniture/test/'.format(MESH_DATA_PATH)
ws_root = '{}/Shapenet_furniture/workspaces_new/'.format(MESH_DATA_PATH)
all_files = os.listdir(root)

z_axis = (0, 0, 1)

if not os.path.exists(ws_root):
    os.mkdir(ws_root)

for single_obj in all_files:
    obj_name = single_obj.split('.')[0]

    bpy.ops.import_scene.obj(filepath=root+single_obj)
    obj = bpy.data.objects[obj_name]

    bm = bmesh.new()
    bm.from_mesh(bpy.data.objects[obj_name].data)

    # Select some faces in the bmesh
    for face in bm.faces:
        face.select = True

    # Update the mesh from the bmesh object
    bpy.context.scene.objects.active = obj
    bm.to_mesh(bpy.context.active_object.data)
    bpy.context.active_object.data.update()

    print("Number of faces in the bmesh:", len(bm.faces))
    print("Number of faces in the original mesh:", len(bpy.context.active_object.data.polygons))

    # Get all the faces that are perpendicular to the z-axis
    z_faces = []
    z_values = []
    for face in bm.faces:
        find_z_faces(z_faces, face, z_values)
    z_values = set(z_values)

    print("Z values: {}".format(z_values))

    z_faces_dict = {'{:0.2f}'.format(z): [] for z in z_values}

    for face in z_faces:
        z = []
        for v in face.verts:
            z.append(v.co.z)
        z_faces_dict['{:0.2f}'.format(np.array(z).mean())].append(face)

    # Combine adjacent regions of z-faces into a single region
    regions = group_adjacent_faces(z_faces)

    print("Found {} regions".format(len(regions)))
    # Approximate the edge of each region as a polygon
    with open('{}/{}_workspace_points.csv'.format(ws_root, obj_name), 'w') as f:
        for cnt, region in enumerate(regions):
            f.write("Region: #{} \n".format(cnt))
            points = []
            for face in region:
                for vertex in face.verts:
                    f.write("{}, {}, {}\n".format(vertex.co.x, vertex.co.y, vertex.co.z))
            f.write("\n")
    bm.free()
