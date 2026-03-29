from delaunay import edge

#private helper functions
def _edge_key(v1, v2):
    return tuple(sorted((v1, v2), key=lambda v: (v.x, v.y, v.z)))

def _unique_vertices(vertices):
    unique = []
    seen = set()
    for v in vertices:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def _order_tetrahedra_around_edge(incident_tetrahedra, edge_vertices):
    """
    Order incident tetrahedra by walking across faces that contain the Delaunay
    edge. For interior edges this yields a cycle; for hull edges it yields an
    open chain.
    """

    if len(incident_tetrahedra) <= 1:
        return list(incident_tetrahedra)

    neighbors = {tetrahedron: [] for tetrahedron in incident_tetrahedra}
    incident_set = set(incident_tetrahedra)

    for tetrahedron in incident_tetrahedra:
        for face in tetrahedron.faces:
            if edge_vertices[0] not in face.vertices or edge_vertices[1] not in face.vertices:
                continue
            if face.adjacent_face is None:
                continue
            adjacent_tetrahedron = face.adjacent_face.tetrahedron
            if adjacent_tetrahedron in incident_set and adjacent_tetrahedron not in neighbors[tetrahedron]:
                neighbors[tetrahedron].append(adjacent_tetrahedron)

    start = next((tet for tet, adjacent in neighbors.items() if len(adjacent) <= 1), incident_tetrahedra[0])

    ordered = []
    visited = set()
    previous = None
    current = start

    while current is not None and current not in visited:
        ordered.append(current)
        visited.add(current)

        next_tetrahedron = None
        for candidate in neighbors[current]:
            if candidate is not previous:
                next_tetrahedron = candidate
                break

        previous, current = current, next_tetrahedron

    if len(ordered) != len(incident_tetrahedra):
        for tetrahedron in incident_tetrahedra:
            if tetrahedron not in visited:
                ordered.append(tetrahedron)

    return ordered

# Main function
def extract_voronoi_from_delaunay(tetrahedra):
    """
    Extract the finite pieces of the Voronoi diagram dual to a Delaunay
    tetrahedralization.

    Returns a dictionary with:
    - "vertices": list[vertex]
    - "edges": list[edge]
    - "faces": dict[(vertex, vertex), list[vertex]]
      Each key is a Delaunay edge and each value is the ordered list of Voronoi
      vertices dual to the tetrahedra incident to that edge.
    - "cells": dict[vertex, dict]
      Each key is a Delaunay site. Each value stores the site's Voronoi cell as
      {"faces": list[list[vertex]], "vertices": list[vertex]}.
    """

    if not tetrahedra:
        return {"vertices": [], "edges": [], "faces": {}, "cells": {}}

    tetrahedron_to_vertex = {}
    voronoi_vertices = []
    voronoi_edges = set()
    delaunay_edge_to_tetrahedra = {}
    delaunay_vertex_to_tetrahedra = {}

    # Vertex: dual of each tetrahedron is its circumcenter.
    for tetrahedron in tetrahedra:
        circumcenter = tetrahedron.circumsphere_center
        tetrahedron_to_vertex[tetrahedron] = circumcenter
        voronoi_vertices.append(circumcenter)

        for site in tetrahedron.vertices:
            delaunay_vertex_to_tetrahedra.setdefault(site, []).append(tetrahedron)

        for i in range(4):
            for j in range(i + 1, 4):
                key = _edge_key(tetrahedron.vertices[i], tetrahedron.vertices[j])
                delaunay_edge_to_tetrahedra.setdefault(key, []).append(tetrahedron)

    # Edge: dual to each shared triangular face.
    for tetrahedron in tetrahedra:
        center = tetrahedron_to_vertex[tetrahedron]
        for face in tetrahedron.faces:
            if face.adjacent_face is None:
                continue
            adjacent_tetrahedron = face.adjacent_face.tetrahedron
            if adjacent_tetrahedron not in tetrahedron_to_vertex:
                continue
            adjacent_center = tetrahedron_to_vertex[adjacent_tetrahedron]
            voronoi_edges.add(edge(center, adjacent_center))

    # Face: collect and order the dual circumcenters around each Delaunay edge.
    voronoi_faces = {}
    for delaunay_edge, incident_tetrahedra in delaunay_edge_to_tetrahedra.items():
        ordered_tetrahedra = _order_tetrahedra_around_edge(incident_tetrahedra, delaunay_edge)
        face_vertices = _unique_vertices(
            [tetrahedron_to_vertex[tetrahedron] for tetrahedron in ordered_tetrahedra]
        )
        if len(face_vertices) >= 2:
            voronoi_faces[delaunay_edge] = face_vertices

            # An open chain still contributes its finite edge segments.
            for i in range(len(face_vertices) - 1):
                voronoi_edges.add(edge(face_vertices[i], face_vertices[i + 1]))

            # Closed cycles correspond to bounded Voronoi faces.
            if len(face_vertices) >= 3:
                first_tetrahedron = ordered_tetrahedra[0]
                last_tetrahedron = ordered_tetrahedra[-1]
                if len(ordered_tetrahedra) > 2:
                    neighbors_of_last = 0
                    for face in last_tetrahedron.faces:
                        if delaunay_edge[0] in face.vertices and delaunay_edge[1] in face.vertices:
                            if (
                                face.adjacent_face is not None
                                and face.adjacent_face.tetrahedron is first_tetrahedron
                            ):
                                neighbors_of_last += 1
                    if neighbors_of_last > 0:
                        voronoi_edges.add(edge(face_vertices[-1], face_vertices[0]))

    # Polyhedron: for each Delaunay site, gather the dual faces of its incident
    # Delaunay edges and collect the circumcenters that bound the cell.
    voronoi_cells = {}
    for site, incident_tetrahedra in delaunay_vertex_to_tetrahedra.items():
        incident_faces = []
        incident_vertices = []

        adjacent_sites = set()
        for tetrahedron in incident_tetrahedra:
            for other_site in tetrahedron.vertices:
                if other_site != site:
                    adjacent_sites.add(other_site)

        for adjacent_site in adjacent_sites:
            dual_face = voronoi_faces.get(_edge_key(site, adjacent_site))
            if dual_face:
                incident_faces.append(dual_face)
                incident_vertices.extend(dual_face)

        voronoi_cells[site] = {
            "faces": incident_faces,
            "vertices": _unique_vertices(incident_vertices),
        }

    return {
        "vertices": _unique_vertices(voronoi_vertices),
        "edges": list(voronoi_edges),
        "faces": voronoi_faces,
        "cells": voronoi_cells,
    }
