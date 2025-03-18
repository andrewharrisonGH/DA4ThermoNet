x_min, x_max, y_min, y_max, z_min, z_max = -8, 8, -8, 8, -8, 8
vertices = [(x, y, z) for x, y, z in [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min), (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max)]]
cmd.create("armstrong_box", "none"); [cmd.pseudoatom("armstrong_box", pos=vertex, name=f"vertex_{i+1}") for i, vertex in enumerate(vertices)]
bonds = [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8)]
[cmd.bond(f"armstrong_box and name vertex_{bond[0]}", f"armstrong_box and name vertex_{bond[1]}") for bond in bonds]
cmd.set("stick_radius", 0.05, "armstrong_box"); cmd.show("sticks", "armstrong_box")