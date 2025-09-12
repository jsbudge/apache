import pickle
from simulib.mesh_functions import getSceneFig, drawOctreeBox
from pathlib import Path
import numpy as np
import plotly.io as pio
import matplotlib as mplib
mplib.use('TkAgg')
from simulib.mesh_objects import TriangleMesh, Scene, VTCMesh
from simulib.mesh_functions import loadTarget

pio.renderers.default = 'browser'

if __name__ == '__main__':
    target_fnme = '/home/jeff/Documents/target_meshes/hangar.targ'
    print(target_fnme)
    scene = Scene()
    # Check if this is some VTC data and load the mesh data accordingly
    if Path(target_fnme).suffix == '.dat':
        mesh = VTCMesh(target_fnme)
    else:
        with open(target_fnme, 'r') as f:
            tdata = f.readlines()
            target_scaling = 1.  # This should be changed if necessary to get the mesh vertices into meter spacing



        print('Loading mesh...', end='')
        source_mesh, mesh_materials = loadTarget(target_fnme)
        source_mesh = source_mesh.rotate(source_mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0]))).scale(target_scaling,
                                                                                                 center=source_mesh.get_center())
        source_mesh = source_mesh.translate(np.array([0, 0, 0.]), relative=False)
        mesh = TriangleMesh(
            source_mesh,
            max_tris_per_split=64,
            material_sigma=[mesh_materials[mtid][1] if mtid in mesh_materials.keys() else mesh_materials[0][1] for mtid in
                            range(np.asarray(source_mesh.triangle_material_ids).max() + 1)],
            material_emissivity=[mesh_materials[mtid][0] if mtid in mesh_materials.keys() else mesh_materials[0][0] for mtid in
                                 range(np.asarray(source_mesh.triangle_material_ids).max() + 1)],
        )
    scene.add(mesh)

    model_name = f'/home/jeff/repo/apache/data/target_meshes/{Path(target_fnme).stem}.model'

    with open(model_name, 'wb') as f:
        pickle.dump(scene, f)

    with open(model_name, 'rb') as f:
        scene = pickle.load(f)

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Choose a colormap
    cmap = cm.get_cmap('viridis')

    # Normalize the data
    tri_materials = np.asarray(source_mesh.triangle_material_ids)
    norm = mcolors.Normalize(vmin=np.min(tri_materials), vmax=np.max(tri_materials))

    # Map data to colors
    rgba_colors = cmap(norm(tri_materials))


    fig = getSceneFig(scene, triangle_colors=rgba_colors[:, :3], title='Depth', zrange=zranges)

    for mesh in scene.meshes:
        d = mesh.bvh_levels - 1
        for b in mesh.bvh[sum(2 ** n for n in range(d)):sum(2 ** n for n in range(d + 1))]:
            if np.sum(b) != 0:
                fig.add_trace(drawOctreeBox(b))
    fig.show()