center_coords = (0,0,0)
half_size = 8 
x_min, x_max, y_min, y_max, z_min, z_max = center_coords[0] - half_size, center_coords[0] + half_size, center_coords[1] - half_size, center_coords[1] + half_size, center_coords[2] - half_size, center_coords[2] + half_size
selection_string = f"((x > {x_min}) and (x < {x_max})) and ((y > {y_min}) and (y < {y_max})) and ((z > {z_min}) and (z < {z_max}))"
cmd.select("cropped_region", selection_string); cmd.select("outside_cropped", "not cropped_region")
cmd.hide("everything","all")
cmd.show("cartoon"   ,"cropped_region")
cmd.show("lines"     ,"center_residues")