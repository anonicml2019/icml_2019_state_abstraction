PUDDLE = [(0.1, 0.8, 0.55, 0.6), (0.3, 0.6, 0.55, 0.4)]

# PUDDLE INFO.
top_puddle = PUDDLE[0]
right_puddle = PUDDLE[1]

# Top Puddle.
pud_out_left_x = min(top_puddle[0], top_puddle[2])
pud_out_top_y = max(top_puddle[1], top_puddle[3])
pud_in_bottom_y = min(top_puddle[1], top_puddle[3])

# Right Puddle.
pud_out_right_x = max(right_puddle[0], right_puddle[2])
pud_out_bottom_y = min(right_puddle[1], right_puddle[3])
pud_in_left_x = min(right_puddle[0], right_puddle[2])