from delaunay import construct_delaunay
from voronoi import extract_voronoi_from_delaunay
import numpy as np
try:
    import pyvista as pv
except ModuleNotFoundError:
    pv = None

def generate_test_points(num_points, x_range=(0, 100), y_range=(0, 100), z_range=(0, 100)):
    import random
    points = []
    for _ in range(num_points):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)
        points.append((x, y, z))
    return points


def primitive_basis_from_lengths_angles(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    if np.isclose(np.sin(gamma), 0.0):
        raise ValueError("gamma must not be 0 or 180 degrees")

    v1 = np.array([a, 0.0, 0.0], dtype=float)
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0], dtype=float)

    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c**2 - cx**2 - cy**2
    if cz_sq < -1e-10:
        raise ValueError("invalid lattice parameters")
    cz = np.sqrt(max(cz_sq, 0.0))
    v3 = np.array([cx, cy, cz], dtype=float)

    basis = np.vstack((v1, v2, v3))
    if np.isclose(np.linalg.det(basis), 0.0):
        raise ValueError("lattice basis must have non-zero volume")
    return basis


def normalize_bravais_lattice_name(lattice_type):
    key = lattice_type.lower().replace("-", " ").replace("_", " ").strip()
    key = " ".join(key.split())

    aliases = {
        "triclinic": "triclinic primitive",
        "triclinic primitive": "triclinic primitive",
        "ap": "triclinic primitive",
        "monoclinic": "monoclinic primitive",
        "monoclinic primitive": "monoclinic primitive",
        "mp": "monoclinic primitive",
        "base centered monoclinic": "monoclinic base centered",
        "monoclinic base centered": "monoclinic base centered",
        "c centered monoclinic": "monoclinic base centered",
        "mc": "monoclinic base centered",
        "orthorhombic": "orthorhombic primitive",
        "orthorhombic primitive": "orthorhombic primitive",
        "op": "orthorhombic primitive",
        "base centered orthorhombic": "orthorhombic base centered",
        "orthorhombic base centered": "orthorhombic base centered",
        "c centered orthorhombic": "orthorhombic base centered",
        "oc": "orthorhombic base centered",
        "body centered orthorhombic": "orthorhombic body centered",
        "orthorhombic body centered": "orthorhombic body centered",
        "i centered orthorhombic": "orthorhombic body centered",
        "oi": "orthorhombic body centered",
        "face centered orthorhombic": "orthorhombic face centered",
        "orthorhombic face centered": "orthorhombic face centered",
        "f centered orthorhombic": "orthorhombic face centered",
        "of": "orthorhombic face centered",
        "tetragonal": "tetragonal primitive",
        "tetragonal primitive": "tetragonal primitive",
        "tp": "tetragonal primitive",
        "body centered tetragonal": "tetragonal body centered",
        "tetragonal body centered": "tetragonal body centered",
        "it": "tetragonal body centered",
        "hexagonal": "hexagonal",
        "hexagonal primitive": "hexagonal",
        "hp": "hexagonal",
        "rhombohedral": "rhombohedral",
        "trigonal": "rhombohedral",
        "hr": "rhombohedral",
        "cubic": "cubic primitive",
        "cubic primitive": "cubic primitive",
        "simple cubic": "cubic primitive",
        "sc": "cubic primitive",
        "body centered cubic": "cubic body centered",
        "cubic body centered": "cubic body centered",
        "bcc": "cubic body centered",
        "face centered cubic": "cubic face centered",
        "cubic face centered": "cubic face centered",
        "fcc": "cubic face centered",
    }

    if key not in aliases:
        raise ValueError(f"unknown Bravais lattice type: {lattice_type}")
    return aliases[key]


def get_bravais_lattice_basis(lattice_type, a=1.0, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0):
    lattice_type = normalize_bravais_lattice_name(lattice_type)

    if b is None:
        b = a
    if c is None:
        c = a

    if lattice_type == "triclinic primitive":
        return primitive_basis_from_lengths_angles(a, b, c, alpha, beta, gamma)
    if lattice_type == "monoclinic primitive":
        return primitive_basis_from_lengths_angles(a, b, c, 90.0, beta, 90.0)
    if lattice_type == "monoclinic base centered":
        return np.array([
            [a / 2.0, 0.0, -c / 2.0],
            [0.0, b, 0.0],
            [a / 2.0, 0.0, c / 2.0],
        ], dtype=float)
    if lattice_type == "orthorhombic primitive":
        return np.array([
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, c],
        ], dtype=float)
    if lattice_type == "orthorhombic base centered":
        return np.array([
            [a / 2.0, -b / 2.0, 0.0],
            [a / 2.0, b / 2.0, 0.0],
            [0.0, 0.0, c],
        ], dtype=float)
    if lattice_type == "orthorhombic body centered":
        return np.array([
            [-a / 2.0, b / 2.0, c / 2.0],
            [a / 2.0, -b / 2.0, c / 2.0],
            [a / 2.0, b / 2.0, -c / 2.0],
        ], dtype=float)
    if lattice_type == "orthorhombic face centered":
        return np.array([
            [0.0, b / 2.0, c / 2.0],
            [a / 2.0, 0.0, c / 2.0],
            [a / 2.0, b / 2.0, 0.0],
        ], dtype=float)
    if lattice_type == "tetragonal primitive":
        return np.array([
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, c],
        ], dtype=float)
    if lattice_type == "tetragonal body centered":
        return np.array([
            [-a / 2.0, a / 2.0, c / 2.0],
            [a / 2.0, -a / 2.0, c / 2.0],
            [a / 2.0, a / 2.0, -c / 2.0],
        ], dtype=float)
    if lattice_type == "hexagonal":
        return np.array([
            [a, 0.0, 0.0],
            [a / 2.0, np.sqrt(3.0) * a / 2.0, 0.0],
            [0.0, 0.0, c],
        ], dtype=float)
    if lattice_type == "rhombohedral":
        return primitive_basis_from_lengths_angles(a, a, a, alpha, alpha, alpha)
    if lattice_type == "cubic primitive":
        return np.array([
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
        ], dtype=float)
    if lattice_type == "cubic body centered":
        return np.array([
            [-a / 2.0, a / 2.0, a / 2.0],
            [a / 2.0, -a / 2.0, a / 2.0],
            [a / 2.0, a / 2.0, -a / 2.0],
        ], dtype=float)
    if lattice_type == "cubic face centered":
        return np.array([
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        ], dtype=float)

    raise ValueError(f"unsupported Bravais lattice type: {lattice_type}")


def reciprocal_basis_from_direct_basis(direct_basis):
    a1, a2, a3 = np.asarray(direct_basis, dtype=float)
    volume = np.dot(a1, np.cross(a2, a3))
    if np.isclose(volume, 0.0):
        raise ValueError("direct lattice basis must have non-zero volume")

    b1 = 2.0 * np.pi * np.cross(a2, a3) / volume
    b2 = 2.0 * np.pi * np.cross(a3, a1) / volume
    b3 = 2.0 * np.pi * np.cross(a1, a2) / volume
    return np.vstack((b1, b2, b3))


def generate_lattice_points_from_basis(basis, index_range=1):
    basis = np.asarray(basis, dtype=float)

    if isinstance(index_range, int):
        ranges = [range(-index_range, index_range + 1) for _ in range(3)]
    else:
        if len(index_range) != 3:
            raise ValueError("index_range must be an int or a length-3 iterable")
        ranges = [range(-r, r + 1) for r in index_range]

    points = []
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                point = i * basis[0] + j * basis[1] + k * basis[2]
                points.append((point[0], point[1], point[2]))
    return points


def perturb_points(points, eps=1e-8):
    perturbed_points = []
    for i, point in enumerate(points):
        dx = eps * (i + 1)
        dy = eps * (i + 1) ** 2
        dz = eps * (i + 1) ** 3
        perturbed_points.append((point[0] + dx, point[1] + dy, point[2] + dz))
    return perturbed_points


def generate_reciprocal_lattice(lattice_type, index_range=1, a=1.0, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0):
    direct_basis = get_bravais_lattice_basis(
        lattice_type,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    reciprocal_basis = reciprocal_basis_from_direct_basis(direct_basis)
    reciprocal_points = generate_lattice_points_from_basis(reciprocal_basis, index_range=index_range)
    reciprocal_points = perturb_points(reciprocal_points)
    return reciprocal_points, direct_basis, reciprocal_basis


def generate_brillouin_zone_example(lattice_type="cubic primitive", index_range=2, a=1.0, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0):
    points, direct_basis, reciprocal_basis = generate_reciprocal_lattice(
        lattice_type,
        index_range=index_range,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return points, direct_basis, reciprocal_basis


def get_brillouin_zone_cell(voronoi_diagram, tol=1e-9):
    origin_cell = None
    origin_site = None

    for site, cell in voronoi_diagram["cells"].items():
        if abs(site.x) < tol and abs(site.y) < tol and abs(site.z) < tol:
            origin_site = site
            origin_cell = cell
            break

    return origin_site, origin_cell



def tetrahedra_to_pyvista_grid(tetrahedra):
    """
    Converts your list[tetrahedron] into a pyvista.UnstructuredGrid.
    Assumes each tetrahedron has .vertices = [v1,v2,v3,v4] where each v has x,y,z.
    """

    if pv is None:
        raise ModuleNotFoundError("pyvista is required for visualization")

    # Map each vertex object to a unique point index.
    # Your vertex dataclass is hashable, so this works as long as you're reusing vertex objects consistently.
    vid = {}
    points = []

    def get_vid(v):
        if v not in vid:
            vid[v] = len(points)
            points.append([v.x, v.y, v.z])
        return vid[v]

    # VTK cell format: [n_points, i0, i1, i2, i3] per tetra
    cells = []
    celltypes = []

    VTK_TETRA = 10  # vtk.VTK_TETRA

    for t in tetrahedra:
        ids = [get_vid(v) for v in t.vertices]
        cells.extend([4, ids[0], ids[1], ids[2], ids[3]])
        celltypes.append(VTK_TETRA)

    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    celltypes = np.asarray(celltypes, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, points)
    # One scalar per tetra so we can color each cell differently.
    grid.cell_data["tet_id"] = np.arange(len(tetrahedra), dtype=np.int32)
    return grid


def normalize_input_points(input_points):
    pts = np.asarray(input_points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("input_points must be an (n,2) or (n,3) array-like")
    if pts.shape[1] == 2:
        pts = np.column_stack((pts, np.zeros(len(pts), dtype=float)))
    elif pts.shape[1] != 3:
        raise ValueError("input_points must have 2 or 3 columns")
    return pts

def plot_delaunay_wireframe(tetrahedra, input_points=None):
    if pv is None:
        raise ModuleNotFoundError("pyvista is required for visualization")
    grid = tetrahedra_to_pyvista_grid(tetrahedra)
    edges = grid.extract_all_edges()

    pl = pv.Plotter()
    pl.add_mesh(
        grid,
        scalars="tet_id",
        show_edges=False,
        opacity=0.35,
        cmap="tab20",
        show_scalar_bar=False,
    )
    # Overlay every tetra edge (including interior shared edges).
    pl.add_mesh(
        edges,
        color="black",
        line_width=3.0,
        render_lines_as_tubes=True,
    )

    if input_points is not None:
        pts = normalize_input_points(input_points)

        pl.add_points(
            pts,
            color="yellow",
            point_size=12,
            render_points_as_spheres=True,
        )

    pl.add_axes()
    pl.show_grid()
    pl.show()


def voronoi_to_pyvista_mesh(voronoi_diagram):
    if pv is None:
        raise ModuleNotFoundError("pyvista is required for visualization")
    points = []
    vid = {}

    def get_vid(v):
        if v not in vid:
            vid[v] = len(points)
            points.append([v.x, v.y, v.z])
        return vid[v]

    lines = []
    for e in voronoi_diagram["edges"]:
        lines.extend([2, get_vid(e.v1), get_vid(e.v2)])

    if not points or not lines:
        return None

    bounded_faces = []
    edge_pairs = {frozenset((e.v1, e.v2)) for e in voronoi_diagram["edges"]}

    for face_vertices in voronoi_diagram["faces"].values():
        if len(face_vertices) < 3:
            continue
        if frozenset((face_vertices[-1], face_vertices[0])) not in edge_pairs:
            continue
        face_ids = [get_vid(v) for v in face_vertices]
        for i in range(1, len(face_ids) - 1):
            bounded_faces.extend([3, face_ids[0], face_ids[i], face_ids[i + 1]])

    points = np.asarray(points, dtype=float)
    line_data = np.asarray(lines, dtype=np.int64)

    line_mesh = pv.PolyData(points, lines=line_data)
    surface_mesh = None

    if bounded_faces:
        face_data = np.asarray(bounded_faces, dtype=np.int64)
        surface_mesh = pv.PolyData(points, faces=face_data)
        surface_mesh.cell_data["face_id"] = np.arange(surface_mesh.n_cells, dtype=np.int32)

    return surface_mesh, line_mesh


def voronoi_cell_to_pyvista_mesh(cell):
    if pv is None:
        raise ModuleNotFoundError("pyvista is required for visualization")
    if cell is None or not cell["faces"]:
        return None

    points = []
    vid = {}

    def get_vid(v):
        if v not in vid:
            vid[v] = len(points)
            points.append([v.x, v.y, v.z])
        return vid[v]

    faces = []
    lines = []
    seen_edges = set()

    for face_vertices in cell["faces"]:
        if len(face_vertices) < 3:
            continue

        face_ids = [get_vid(v) for v in face_vertices]
        for i in range(1, len(face_ids) - 1):
            faces.extend([3, face_ids[0], face_ids[i], face_ids[i + 1]])

        for i in range(len(face_ids)):
            j = (i + 1) % len(face_ids)
            edge_key = tuple(sorted((face_ids[i], face_ids[j])))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                lines.extend([2, face_ids[i], face_ids[j]])

    if not points:
        return None

    points = np.asarray(points, dtype=float)
    line_mesh = None
    surface_mesh = None

    if lines:
        line_mesh = pv.PolyData(points, lines=np.asarray(lines, dtype=np.int64))
    if faces:
        surface_mesh = pv.PolyData(points, faces=np.asarray(faces, dtype=np.int64))
        surface_mesh.cell_data["face_id"] = np.arange(surface_mesh.n_cells, dtype=np.int32)

    return surface_mesh, line_mesh


def plot_voronoi_diagram(voronoi_diagram, input_points=None):
    if pv is None:
        raise ModuleNotFoundError("pyvista is required for visualization")
    meshes = voronoi_to_pyvista_mesh(voronoi_diagram)
    if meshes is None:
        print("No Voronoi edges produced; skipping plot.")
        return
    surface_mesh, line_mesh = meshes

    pl = pv.Plotter()
    if surface_mesh is not None:
        pl.add_mesh(
            surface_mesh,
            scalars="face_id",
            show_edges=False,
            opacity=0.35,
            cmap="tab20",
            show_scalar_bar=False,
        )

    pl.add_mesh(
        line_mesh,
        color="black",
        line_width=3.0,
        render_lines_as_tubes=True,
    )

    used_vertices = []
    seen = set()
    for e in voronoi_diagram["edges"]:
        for v in (e.v1, e.v2):
            if v not in seen:
                seen.add(v)
                used_vertices.append(v)

    voronoi_points = np.asarray([[v.x, v.y, v.z] for v in used_vertices], dtype=float)
    if len(voronoi_points) > 0:
        pl.add_points(
            voronoi_points,
            color="white",
            point_size=10,
            render_points_as_spheres=True,
        )

    if input_points is not None:
        pts = normalize_input_points(input_points)

        pl.add_points(
            pts,
            color="yellow",
            point_size=12,
            render_points_as_spheres=True,
        )

    pl.add_axes()
    pl.show_grid()
    pl.show()


def plot_brillouin_zone(voronoi_diagram, reciprocal_points=None):
    if pv is None:
        raise ModuleNotFoundError("pyvista is required for visualization")

    origin_site, cell = get_brillouin_zone_cell(voronoi_diagram)
    if cell is None:
        print("Could not find the Voronoi cell of the origin; skipping Brillouin zone plot.")
        return

    meshes = voronoi_cell_to_pyvista_mesh(cell)
    if meshes is None:
        print("No bounded Brillouin zone geometry produced; skipping plot.")
        return
    surface_mesh, line_mesh = meshes

    pl = pv.Plotter()
    if surface_mesh is not None:
        pl.add_mesh(
            surface_mesh,
            scalars="face_id",
            show_edges=False,
            opacity=0.35,
            cmap="tab20",
            show_scalar_bar=False,
        )

    if line_mesh is not None:
        pl.add_mesh(
            line_mesh,
            color="black",
            line_width=3.0,
            render_lines_as_tubes=True,
        )

    if reciprocal_points is not None:
        pts = normalize_input_points(reciprocal_points)
        pl.add_points(
            pts,
            color="yellow",
            point_size=10,
            render_points_as_spheres=True,
        )

    if origin_site is not None:
        pl.add_points(
            np.asarray([[origin_site.x, origin_site.y, origin_site.z]], dtype=float),
            color="red",
            point_size=14,
            render_points_as_spheres=True,
        )

    pl.add_axes()
    pl.show_grid()
    pl.show()

if __name__ == "__main__":
    # Generate a reciprocal lattice for a Brillouin zone example
    points, direct_basis, reciprocal_basis = generate_brillouin_zone_example(
        "tetragonal body centered",
        index_range=1,
        a=1.0,
    )
    print("Direct basis:")
    print(direct_basis)
    print("Reciprocal basis:")
    print(reciprocal_basis)

    # Generate some random points for testing
    # points = generate_test_points(35)
    # points = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 5), (5, 5, 5), (10, 10, 10)]
    # points = [(0,0,0)]
    # points = [(0,0,0), (2, 2, 2)]
    # points = [
    #     [0.0, 0.0, 0.0],
    #     [2.0, 0.0, 0.0],
    #     [0.0, 3.0, 0.0],
    #     [0.0, 0.0, 4.0],
    # ]

    # Construct the Delaunay triangulation (tetrahedralization)
    tetrahedra = construct_delaunay(points)

    # Plot the resulting tetrahedra as a wireframe
    if not tetrahedra:
        print("No tetrahedra produced; skipping plot.")
    else:
        plot_delaunay_wireframe(tetrahedra, points)
        voronoi_diagram = extract_voronoi_from_delaunay(tetrahedra)
        plot_voronoi_diagram(voronoi_diagram, points)
        plot_brillouin_zone(voronoi_diagram, points)
