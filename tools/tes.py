from line import Line

A = (57, 63)
B = (157, 60)
C = (156, 60)
D = (314, 63)

angle_tolerance_deg=2.0
max_y_deviation=20

line = Line()

result = line.is_straight( A, B, C, D )
print(result)

