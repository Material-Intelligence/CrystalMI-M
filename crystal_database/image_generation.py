# -*- coding: utf-8 -*-
"""
    This module defines the polygons.
    It requires modules: crystal_class, basis, CrystalLattice,
    ps_setd_mingled, mlab, tvtk and enclose_polyhedron.
"""
import numpy as np
import nlopt

from utils.crystal_lattice import CrystalLattice
from utils.cgeometry import enclosure
import utils.crystal_class as CC
from utils.crystal_shape import ps_setd_mingled, enclose_polyhedron
import utils.basis as basis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import pandas as pd

def clear_directory(directory):
    """
    Delete all files in the specified directory.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
def crystal_plane(hkl):
    '''
       return the geometric planes of the polyhedron, where the 32 point group is considered,
       and the lattice parameter is taken as 0.4nm.
       hkl: Miller indices of crystallographic planes.
    '''
    b1 = basis.cubic(7.3253)  # unit: nm
    cc1 = CC.CrystalClass30()
    latt = CrystalLattice(b1, cc1)
    planes = []
    number_planes = []
    for index in hkl:
        pf = latt.geometric_plane_family(index)
        number_planes.append(len(pf))
        planes.extend(pf)
    return planes, number_planes


def plot_crystal(qfp, number_planes, surf_color, surf_alpha, edge_color, edge_width, filename):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(number_planes)):
        start = sum(number_planes[:i])
        end = sum(number_planes[:i + 1])
        for surf_index in range(start, end):
            if qfp[2][surf_index]:  # Ensure the face has vertices
                vertices = qfp[2][surf_index]
                if len(vertices) > 0:  # Check if the vertex is empty
                    poly = Poly3DCollection([vertices], alpha=1.0)
                    poly.set_facecolor(surf_color[i])
                    poly.set_edgecolor(edge_color)
                    poly.set_linewidth(edge_width)
                    ax.add_collection3d(poly)

    all_points = [np.array(v) for v in qfp[2] if len(v) > 0]  # Only non-empty vertex sets are kept
    if all_points:
        all_points = np.concatenate(all_points)
        ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
        ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
        ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=20, azim=45)
    ax.set_axis_off()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    print("Image saved as %s" % filename)


if __name__ == "__main__":
    # Generate folder name based on parameters
    b1 = basis.cubic(0.4)
    cc1 = CC.CrystalClass32()
    hkl = [[1, 1, 1], [1, 1, 2]]

    cubic_str = "cubic"
    cc_str = "m-3m"

    # Reading data from Excel files
    csv_file = 'database/cubic_m-3m/2/cubic_m-3m_111112.csv'
    surface_energy_df = pd.read_csv(csv_file)

    energy100_values = surface_energy_df['energy111'].values
    energy111_values = surface_energy_df['energy112'].values

    output_folder = "image_temp"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_directory(output_folder)

    planes, number_planes = crystal_plane(hkl)

    surf_color = [(0.93, 0.26, 0.25), (0.36, 0.78, 0.85), (0.96, 0.58, 0.25), (0.08, 0.65, 0.25), (0.44, 0.38, 0.65)]
    surf_alpha = [0.8, 0.6]  # Set the transparency of each face family
    edge_color = 'White'  # Set the edge color
    edge_width = 1  # Set the width of the edge

    image_count = 0
    for energy100, energy111 in zip(energy100_values, energy111_values):
        if image_count >= 10:
            break

        energy100 = float(energy100)
        energy111 = float(energy111)

        ps_setd_mingled(number_planes, planes, [energy100, energy111])
        diameter = 10.0
        volume = 4.0 / 3.0 * np.pi * (diameter * 0.5) ** 3
        qfp = enclose_polyhedron(number_planes, planes, volume)
        filename = os.path.join(output_folder,
                                "{}_{}_{}_{}.png".format(cubic_str, cc_str, energy100, energy111))

        plot_crystal(qfp, number_planes, surf_color, surf_alpha, edge_color, edge_width, filename)

        image_count += 1
