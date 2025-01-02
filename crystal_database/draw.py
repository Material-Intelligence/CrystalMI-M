import numpy as np
import nlopt
from mayavi import mlab
from tvtk.api import tvtk
import utils.basis as basis
from utils.crystal_lattice import CrystalLattice
from utils.cgeometry import enclosure
import utils.crystal_class as CC
from utils.crystal_shape import ps_setd_mingled, enclose_polyhedron


def crystal_surface(intersection_point):
    surf_point = [len(intersection_point)]  # number of intersections
    for i in np.arange(len(intersection_point)):
        surf_point.append(i)

    return surf_point


def crystal_plane(hkl):
    '''
       return the geometric planes of the polyhedron, where the 32 point group is considered,
	   and the lattice parameter is taken as 0.4nm.
	   hkl: Miller indices of crystallographic planes.
    '''
    b1 = basis.cubic(0.4)  # unit: nm

    cc1 = CC.CrystalClass32()
    latt = CrystalLattice(b1, cc1)
    planes = []
    number_planes = []
    for index in hkl:
        pf = latt.geometric_plane_family(index)
        number_planes.append(len(pf))
        planes.extend(pf)
    return planes, number_planes


surf_dist = [1, 1.43321198634055]
hkl = [[1, 0, 0], [1, 1, 1]]
surf_color = [(0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

diameter = 20.0
volume = 4.0 / 3.0 * np.pi * (diameter * 0.5) ** 3
planes, number_planes = crystal_plane(hkl)
ps_setd_mingled(number_planes, planes, surf_dist)
qfp = enclose_polyhedron(number_planes, planes, volume)
for i in np.arange(len(number_planes)):
    for surf_index in np.arange(sum(number_planes[0:i]), sum(number_planes[0:i + 1])):
        if qfp[2][surf_index] == []:
            break
        else:
            p1 = tvtk.PolyData()
            points_data = qfp[2][surf_index]

            p1.points = qfp[2][surf_index]  # the coordinates of the intersection of each plane.
            a = qfp[2][surf_index]
            faces = crystal_surface(qfp[2][surf_index])
            cells = tvtk.CellArray()  # create a new CellArray object to assign the polys property.
            cells.set_cells(1,
                            faces)  # the first parameter is the number of faces (here is 1),
            p1.polys = cells  # and the second parameter is an array describing the composition of each face.
            p1.point_data.scalars = np.linspace(0.0, 1.0, len(p1.points))
            mlab.figure(number_planes, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.pipeline.surface(p1, representation='surface', opacity=1.0, color=surf_color[i])
axe = tvtk.AxesActor(total_length=(3, 3, 3))
mlab.show()
