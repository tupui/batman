import numpy as np


def vtk_mesh(coordinates, data):

    n_params = 3
    n_points = 6
    # n_points, n_params = coordinates.shape

    n_cell = n_points // n_params // 2

    # Building list of node for a given parameter: X1 -> 0 3 6 9
    base = np.array([i * n_params for i in range(n_cell)])
    base = np.array([base + i for i in range(n_params * 2)]).T

    # Building cells for the first stack
    first_cycle = [i for i in range(n_params)]
    second_cycle = [i for i in range(n_params, n_params * 2)]

    # 01 12 23 30
    first_stack = [[first_cycle[i], first_cycle[i] + 1] for i in range(n_params)]
    second_stack = [[second_cycle[i], second_cycle[i] + 1] for i in range(n_params)]
    first_stack[-1][1] = 0
    second_stack[-1][1] = n_params

    first_cells = np.array(list(zip(first_stack, second_stack))).reshape(-1, 4)

    # Combining base using first_cells indices
    out = [base[:, first_cells[i]] for i in range(n_params)]
    connectivity = np.array(list(zip(*out))).reshape(-1, 4)

    connectivity = np.concatenate((4 * np.ones((connectivity.shape[0], 1)),
                                   connectivity), axis=1)
    connectivity = connectivity.astype(int, copy=False)

    # Write mesh file in VTK ascii format
    with open('mesh_kiviat.vtk', 'wb') as f:
        header = ("# vtk DataFile Version 2.0\n"
                  "Kiviat 3D\n"
                  "ASCII\n\n"
                  "DATASET UNSTRUCTURED_GRID\n\n")
        f.write(header.encode('utf8'))

        # Points coordinates
        np.savetxt(f, coordinates, header='POINTS {} float'.format(n_points),
                   delimiter=' ', footer=' ', comments='')
        # Connectivity
        np.savetxt(f, connectivity, fmt='%d',
                   header='CELLS {} {}'.format(connectivity.shape[0], np.prod(connectivity.shape)),
                   footer=' ', comments='')

        # Cell types
        cell_types = 'CELL_TYPES {}\n'.format(connectivity.shape[0]).encode('utf8')
        f.write(cell_types)
        f.writelines(['8\n'.encode('utf8') for _ in range(connectivity.shape[0])])

        # Point data
        np.savetxt(f, data,
                   header="\nPOINT_DATA {}\n"
                          "SCALARS value double\n"
                          "LOOKUP_TABLE default".format(n_points),
                   footer=' ', comments='')

vtk_mesh(np.array([[1, 3, 5]]), np.array([10]))
