import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

@dataclass(frozen=True, slots=True)
class vertex:
    x: float
    y: float
    z: float


class edge:
    def __init__(self, v1, v2):
        #represents an edge from v1 to v2
        self.v1 = v1
        self.v2 = v2
        self.rface = None
        self.lface = None

    def __eq__(self, other):
        #two edges are equal if they have the same vertices, regardless of order
        if not isinstance(other, edge):
            return NotImplemented
        return (self.v1 == other.v1 and self.v2 == other.v2) or (self.v1 == other.v2 and self.v2 == other.v1)

    def __hash__(self):
        return hash(frozenset((self.v1, self.v2)))
        

#These will be "half-faces" that are used to represent the faces of the tetrahedra. They are half faces because each tetrahedron will need vertices ordered in a specific way to ensure that the orientation predicate works correctly. Each face will have a reference to the other face that is adjacent to it, which allows for quick access to the adjacent tetrahedra when traversing the triangulation.
class face:
    def __init__(self, v1, v2, v3, opposite_vertex):
        orient_val = orient(v1, v2, v3, opposite_vertex)
        while orient_val >= 0:
            #if the orientation is not positive, we need to swap two vertices to ensure that the orientation is correct
            v2, v3 = v3, v2
            if orient_val == 0:
                #if the orientation is 0, then the vertices are coplanar, which should not happen under general position, but we can handle it by simply leaving the vertices as they are and allowing the algorithm to proceed, as it will not cause any issues with the correctness of the algorithm
                break
            orient_val = orient(v1, v2, v3, opposite_vertex)

        self.vertices = [v1, v2, v3]
        self.tetrahedron = None # the tetrahedron that this face belongs to
        self.adjacent_face = None # the face that is adjacent to this face across the edge opposite to the first vertex (v1)

    def __eq__(self, other):
        #two faces are equal if they have the same vertices, regardless of order

        return set(self.vertices) == set(other.vertices)
    
    def __str__(self):
        return f"face({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"

        

class tetrahedron:
    def __init__(self, v1, v2, v3, v4):
        self.vertices = [v1, v2, v3, v4]
        self.edges = [] # for completeness
        self.faces = [] #allows for quick access to the adjacent tetrahedra
        self.circumsphere_center, self.circumsphere_radius = self.circumsphere()
        self.create_faces()
    
    def circumsphere(self):
        #returns the center and radius of the circumsphere of the tetrahedron
        A = np.array([[self.vertices[0].x, self.vertices[0].y, self.vertices[0].z, 1],
                      [self.vertices[1].x, self.vertices[1].y, self.vertices[1].z, 1],
                      [self.vertices[2].x, self.vertices[2].y, self.vertices[2].z, 1],
                      [self.vertices[3].x, self.vertices[3].y, self.vertices[3].z, 1]])
        B = np.array([[self.vertices[0].x**2 + self.vertices[0].y**2 + self.vertices[0].z**2],
                      [self.vertices[1].x**2 + self.vertices[1].y**2 + self.vertices[1].z**2],
                      [self.vertices[2].x**2 + self.vertices[2].y**2 + self.vertices[2].z**2],
                      [self.vertices[3].x**2 + self.vertices[3].y**2 + self.vertices[3].z**2]])
        try:
            C = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # Degenerate tetrahedron (coplanar/collinear points): keep running.
            return vertex(0.0, 0.0, 0.0), np.inf
        center = vertex(C[0][0]/2, C[1][0]/2, C[2][0]/2)
        radius = np.sqrt((center.x - self.vertices[0].x)**2 + (center.y - self.vertices[0].y)**2 + (center.z - self.vertices[0].z)**2)
        return center, radius
    
    def create_faces(self):
        #creates the faces of the tetrahedron and ensures that they are oriented correctly
        self.faces.append(face(self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]))
        self.faces.append(face(self.vertices[0], self.vertices[1], self.vertices[3], self.vertices[2]))
        self.faces.append(face(self.vertices[0], self.vertices[2], self.vertices[3], self.vertices[1]))
        self.faces.append(face(self.vertices[1], self.vertices[2], self.vertices[3], self.vertices[0]))

        for new_face in self.faces:
            new_face.tetrahedron = self

    def __str__(self):
        return f"tetrahedron({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]}, {self.vertices[3]})"
    
class DelaunayTriangulation:
    def __init__(self, points):
        self.points = points
        self.tetrahedra = []

def opposite_vertex(tet, f):
    """
    Returns the vertex of tet not on face f.
    """
    fset = set(f.vertices)
    for v in tet.vertices:
        if v not in fset:
            return v
    return None

def make_super_tetra(points, eps=1e-12):
    """
        Constructs a bounded super tetrahedron that contains all input points.
        Uses a closed-form construction around the axis-aligned bounding box
        (no exponential growth loop).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("make_super_tetra expects points shaped (n, 3)")

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    cx, cy, cz = 0.5 * (mins + maxs)

    # Bounding-box span and a generous safety factor.
    span = max(maxs - mins)
    d = max(1.0, 10.0 * span + eps)

    # Standard super tetra around bbox center.
    v1 = vertex(cx,            cy,            cz + 3.0 * d)
    v2 = vertex(cx - 2.0 * d,  cy - 1.0 * d,  cz - 1.0 * d)
    v3 = vertex(cx + 2.0 * d,  cy - 1.0 * d,  cz - 1.0 * d)
    v4 = vertex(cx,            cy + 2.0 * d,  cz - 1.0 * d)
    return tetrahedron(v1, v2, v3, v4)

#we define two predicates: orient and in_sphere.
def orient(a, b, c, p):
    """
        returns a positive value if p is above the plane defined by a, b, and c, a negative value if p is below the plane, and 0 if p is on the plane
        This is done by constructing a matrix M, then taking its determinant
        Input: a, b, c, p are vertices
        Output: a positive value if p is above the plane defined by a, b, and c, a negative value if p is below the plane, and 0 if p is on the plane
    """
    
    #first we construct the matrix M whose rows are the coordinates of a, b, c, and p, with an additional column of 1s
    M = np.array([[a.x, a.y, a.z, 1],
                  [b.x, b.y, b.z, 1],
                  [c.x, c.y, c.z, 1],
                  [p.x, p.y, p.z, 1]])
    #the orientation is given by the sign of the determinant of M
    return np.linalg.det(M)

def in_sphere(a, b, c, d, p):
    """
        returns a positive value if p is inside the circumsphere of the tetrahedron defined by a, b, c, and d, a negative value if p is outside the circumsphere, and 0 if p is on the circumsphere
        This is done by constructing a matrix M, then taking its determinant
        Input: a, b, c, d, p are vertices
        Output: a positive value if p is inside the circumsphere of the tetrahedron defined by a, b, c, and d, a negative value if p is outside the circumsphere, and 0 if p is on the circumsphere
    """
    #first, construct the matrix M
    M = np.array([[a.x, a.y, a.z, a.x**2 + a.y**2 + a.z**2, 1],
                  [b.x, b.y, b.z, b.x**2 + b.y**2 + b.z**2, 1],
                  [c.x, c.y, c.z, c.x**2 + c.y**2 + c.z**2, 1],
                  [d.x, d.y, d.z, d.x**2 + d.y**2 + d.z**2, 1],
                  [p.x, p.y, p.z, p.x**2 + p.y**2 + p.z**2, 1]])
    
    return np.linalg.det(M)

def find_containing_tetrahedron(triangulation, point):
    """
        Traverses the Delaunay triangulation utilizing a Heuristic to find the tetrahedron that contains the given point. This is done by starting at an arbitrary tetrahedron and checking that orient returns a positive value for all four faces of the tetrahedron. If orient returns a negative value for any face, we move to the adjacent tetrahedron across that face and repeat the process until we find a tetrahedron that contains the point. 

        Input: triangulation is a DelaunayTriangulation object, point is a vertex
        Output: the tetrahedron that contains the point
    """
    
    if not triangulation.tetrahedra:
        return None

    eps = 1e-12
    # Walk-based point location.
    current_tetrahedron = triangulation.tetrahedra[np.random.randint(len(triangulation.tetrahedra))]
    visited = set()
    while True:
        tid = id(current_tetrahedron)
        if tid in visited:
            return None
        visited.add(tid)

        outside_face = None
        for face in current_tetrahedron.faces:
            if orient(*face.vertices, point) > eps:
                outside_face = face
                break

        if outside_face is None:
            return current_tetrahedron
        if outside_face.adjacent_face is None:
            return None
        current_tetrahedron = outside_face.adjacent_face.tetrahedron
        
        

#under general position, there are 2 cases:
# 1. one face of an adjacent tetrahedron is visible from the point
# 2. two faces of an adjacent tetrahedron are visible from the point
def flip_tetrahedra(tetrahedron1, tetrahedron2, triangulation, queue):
    #to determine which case, we must determine if the line segment between the apexes (vertices not common)
    # passes through the face between the two tetrahedra. 
    # we can use the orientation predicate to determine this

    #first, we need to identify the common face between the two tetrahedra and the apexes of each tetrahedron
    common_face = None
    apex1 = None
    apex2 = None
    for face1 in tetrahedron1.faces:
        for face2 in tetrahedron2.faces:
            if face1 == face2:
                common_face = face1
                break
        if common_face is not None:
            break
    if common_face is None:
        return False
    for vertex in tetrahedron1.vertices:
        if vertex not in common_face.vertices:
            apex1 = vertex
            break
    for vertex in tetrahedron2.vertices:
        if vertex not in common_face.vertices:
            apex2 = vertex
            break
    if apex1 is None or apex2 is None:
        return False

    #we test each non shared face of t1 to see if it is visible from the apex of t2
    visible_faces = []
    for face in tetrahedron1.faces:
        if face != common_face:
            if orient(*face.vertices, apex2) > 0:
                visible_faces.append(face)

    if len(visible_faces) == 0:
        #case 1: only the shared face is visible from the apex of t2
        # we must create 3 new tetrahedra by connecting the two apexes
        new_tetrahedron1 = tetrahedron(apex1, common_face.vertices[0], common_face.vertices[1], apex2)
        new_tetrahedron2 = tetrahedron(apex1, common_face.vertices[1], common_face.vertices[2], apex2)
        new_tetrahedron3 = tetrahedron(apex1, common_face.vertices[2], common_face.vertices[0], apex2)

        #since adjacent tetrahedra are connected by their faces, we can simply take external faces of the two original tetrahedra, as they will already be correctly connected to the rest of the triangulation
        for face in (tetrahedron1.faces + tetrahedron2.faces):
            for new_face in (new_tetrahedron1.faces + new_tetrahedron2.faces + new_tetrahedron3.faces):
                if face == new_face:
                    new_face.adjacent_face = face.adjacent_face
                    if new_face.adjacent_face is not None:
                        new_face.adjacent_face.adjacent_face = new_face
                
        
        #now we handle new internal faces.
        for face in (new_tetrahedron1.faces + new_tetrahedron2.faces + new_tetrahedron3.faces):
            for other_face in (new_tetrahedron1.faces + new_tetrahedron2.faces + new_tetrahedron3.faces):
                if face == other_face and face.tetrahedron != other_face.tetrahedron:
                    face.adjacent_face = other_face
                    other_face.adjacent_face = face
        #finally, we remove the original tetrahedra from the triangulation and add the new tetrahedra
        if tetrahedron1 not in triangulation.tetrahedra or tetrahedron2 not in triangulation.tetrahedra:
            return False
        triangulation.tetrahedra.remove(tetrahedron1)
        triangulation.tetrahedra.remove(tetrahedron2)
        triangulation.tetrahedra.append(new_tetrahedron1)
        triangulation.tetrahedra.append(new_tetrahedron2)
        triangulation.tetrahedra.append(new_tetrahedron3)
        #add the new tetrahedra to the queue to check for Delaunay condition
        queue.append(new_tetrahedron1)
        queue.append(new_tetrahedron2)
        queue.append(new_tetrahedron3)
        return True

    elif len(visible_faces) == 1:
        #case 2: one face of t1 are visible from the apex of t2, in addition to the shared face.

        #first, we must get the other tetrahedron that is adjacent to the visible face, which we will call t3
        visible_face = visible_faces[0]
        edge_vertices = [v for v in visible_face.vertices if v != apex1]
        if len(edge_vertices) != 2:
            return False
        # Paper precondition: there must exist tau_b sharing edge (a,b) with tau and tau_a.
        t3 = None
        edge_set = set(edge_vertices)
        for candidate in triangulation.tetrahedra:
            if candidate in (tetrahedron1, tetrahedron2):
                continue
            vset = set(candidate.vertices)
            if edge_set.issubset(vset) and apex1 in vset and apex2 in vset:
                t3 = candidate
                break
        if t3 is None:
            return False
        
        #the two vertices that are shared between all three will be the new apexes
        shared_vertices = []
        for vertex in t3.vertices:
            if vertex in tetrahedron1.vertices and vertex in tetrahedron2.vertices:
                shared_vertices.append(vertex)

        all_vertices = list(set(t3.vertices + tetrahedron1.vertices + tetrahedron2.vertices))
        not_shared = [v for v in all_vertices if v not in shared_vertices]

        # For a valid 3->2 flip neighborhood we need exactly 5 unique vertices:
        # 2 shared (common edge) + 3 non-shared.
        if len(shared_vertices) != 2 or len(not_shared) != 3:
            return False

        #the two new tetrahedra will be connected by the face formed by the three non-shared vertices
        new_tetrahedron1 = tetrahedron(shared_vertices[0], not_shared[0], not_shared[1], not_shared[2])
        new_tetrahedron2 = tetrahedron(shared_vertices[1], not_shared[0], not_shared[1], not_shared[2])

        #repeat like the last case to connect the new tetrahedra to the rest of the triangulation
        for face in (tetrahedron1.faces + tetrahedron2.faces + t3.faces):
            for new_face in (new_tetrahedron1.faces + new_tetrahedron2.faces):
                if face == new_face:
                    new_face.adjacent_face = face.adjacent_face
                    if new_face.adjacent_face is not None:
                        new_face.adjacent_face.adjacent_face = new_face
        for face in (new_tetrahedron1.faces + new_tetrahedron2.faces):
            for other_face in (new_tetrahedron1.faces + new_tetrahedron2.faces):
                if face == other_face and face.tetrahedron != other_face.tetrahedron:
                    face.adjacent_face = other_face
                    other_face.adjacent_face = face
        #finally, we remove the original tetrahedra from the triangulation and add the new tetrahedra
        if (tetrahedron1 not in triangulation.tetrahedra or
            tetrahedron2 not in triangulation.tetrahedra or
            t3 not in triangulation.tetrahedra):
            return False
        triangulation.tetrahedra.remove(tetrahedron1)
        triangulation.tetrahedra.remove(tetrahedron2)
        triangulation.tetrahedra.remove(t3)
        triangulation.tetrahedra.append(new_tetrahedron1)
        triangulation.tetrahedra.append(new_tetrahedron2)
        #add the new tetrahedra to the queue to check for Delaunay condition
        queue.append(new_tetrahedron1)
        queue.append(new_tetrahedron2)
        return True
    else:
        #this should never happen under general position, but we can handle it by simply not flipping the tetrahedra and leaving the triangulation as is
        return False


#the pointset P is a list of vertices, which are stored as 3-tuples
#we will utilize numpy arrays for efficient computation of the Delaunay triangulation, so P is a numpy array of shape (n, 3)
def construct_delaunay(points, show_progress=True):
    #constructs the Delaunay triangulation of the given points
    #returns a list of tetrahedra
    
    #first, construct T_large, the initial tetrahedron that contains all the points
    T_large = make_super_tetra(points)

    #initialize the triangulation with T_large
    triangulation = DelaunayTriangulation(points)
    triangulation.tetrahedra.append(T_large)

    #insert each point into the triangulation
    point_iterator = points
    if show_progress and tqdm is not None:
        point_iterator = tqdm(points, desc="Constructing Delaunay", unit="pt")

    for point in point_iterator:
        #find the containing tetrahedron for the point
        containing_tetrahedron = find_containing_tetrahedron(triangulation, vertex(*point))
        if containing_tetrahedron is None:
            raise RuntimeError(f"Failed to locate containing tetrahedron for point {point}")
        # print("Containing tetrahedron for point", point, "is", containing_tetrahedron)
        #initialze a queue to hold new tetrahedra that need to be checked for the Delaunay condition
        queue = deque()
        #create new tetrahedra by connecting the point to the faces of the containing tetrahedron
        new_tetrahedra = []
        for face in containing_tetrahedron.faces:
            new_tetrahedron = tetrahedron(*face.vertices, vertex(*point))
            for new_face in new_tetrahedron.faces:
                if new_face == face:
                    new_face.adjacent_face = face.adjacent_face
                    if new_face.adjacent_face is not None:
                        new_face.adjacent_face.adjacent_face = new_face
            new_tetrahedra.append(new_tetrahedron)
            queue.append(new_tetrahedron)
        
        #now we connect the new tetrahedra to each other by connecting faces that are the same
        all_faces = []
        for new_tetrahedron in new_tetrahedra:
            all_faces.extend(new_tetrahedron.faces)
            
        for face in all_faces:
            for other_face in all_faces:
                if face == other_face and face.tetrahedron != other_face.tetrahedron:
                    face.adjacent_face = other_face
                    other_face.adjacent_face = face

        # for i, new_tetrahedron in enumerate(new_tetrahedra):
        #     print("New tetrahedron:", i)
        #     for j, face in enumerate(new_tetrahedron.faces):
        #         print("  Face", j, ":", face)
        #         if face.adjacent_face is not None:
        #             print("    Adjacent to:", face.adjacent_face, "in tetrahedron", face.adjacent_face.tetrahedron)
        #         else:
        #             print("    No adjacent face")

        triangulation.tetrahedra.extend(new_tetrahedra)
        #remove the containing tetrahedron from the triangulation
        triangulation.tetrahedra.remove(containing_tetrahedron)
        
        # check the Delaunay condition for the new tetrahedra and flip if necessary
        while queue:
            current_tetrahedron = queue.popleft()
            if current_tetrahedron not in triangulation.tetrahedra:
                continue
            #check to see if the current tetrahedron violates the Delaunay condition with any of its neighbors
            flipped = False
            for face in current_tetrahedron.faces:
                if face.adjacent_face is not None:
                    adjacent_tetrahedron = face.adjacent_face.tetrahedron
                    if adjacent_tetrahedron not in triangulation.tetrahedra:
                        continue
                    # Circumsphere-based local Delaunay check.
                    opposite = opposite_vertex(adjacent_tetrahedron, face)
                    if opposite is None:
                        continue
                    center = current_tetrahedron.circumsphere_center
                    radius = current_tetrahedron.circumsphere_radius
                    if np.isfinite(radius):
                        dist = np.sqrt((center.x - opposite.x)**2 +
                                       (center.y - opposite.y)**2 +
                                       (center.z - opposite.z)**2)
                        if dist < radius - 1e-12:
                            if flip_tetrahedra(current_tetrahedron, adjacent_tetrahedron, triangulation, queue):
                                flipped = True
                                break
                    if flipped:
                        break
            if flipped:
                continue
    #finally, we remove any tetrahedra that contain vertices of T_large, as they are not part of the final triangulation
    final_tetrahedra = []
    
    for tetra in triangulation.tetrahedra:
        if all(vertex not in tetra.vertices for vertex in T_large.vertices):
            final_tetrahedra.append(tetra)
    return final_tetrahedra

        
                        

                        
        
