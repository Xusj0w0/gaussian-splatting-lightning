import numpy as np

# st = input("Input transform matrix, q for quit:\n")
# mat = [float(s) for line in st.splitlines() for s in line.strip().split(" ")]
# T = np.array(mat)
# while st.strip() != "q":
#     mat = [float(s) for line in st.splitlines() for s in line.strip().split(" ")]
#     T = np.array(mat) @ T
# print("=" * 20 + "Rotation" + "=" * 20)
# print(np.array2string(T[:3, :3]))
# print("=" * 20 + "Transform" + "=" * 20)
# print(np.array2string(T[:3, 3].reshape(-1)))

st = """
0.931001 -0.060253 0.360009 -32.820862
0.054672 0.998174 0.025674 -119.222771
-0.360898 -0.004220 0.932596 -9.532154
0.000000 0.000000 0.000000 1.000000
"""

mat = [[float(s) for s in line.strip().split(" ")] for line in st.strip().splitlines()]
T = np.array(mat)
print("=" * 20 + "Rotation" + "=" * 20)
print(np.array2string(T[:3, :3].reshape(-1), max_line_width=140))
print("=" * 20 + "Transform" + "=" * 20)
print(np.array2string(T[:3, 3].reshape(-1)))
