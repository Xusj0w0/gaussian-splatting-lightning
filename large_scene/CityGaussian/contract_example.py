import numpy as np
import sys

max_val = 10
ord = 2
for i1 in range(1, max_val):
    for j1 in range(1, max_val):
        for k1 in range(1, max_val):
            vector1 = np.array([i1, j1, k1]) / max_val + 1.0
            for i2 in range(i1 + 1, max_val):
                for j2 in range(j1 + 1, max_val):
                    for k2 in range(k1 + 1, max_val):
                        vector2 = np.array([i2, j2, k2]) / max_val + 1.0

                        norm1, norm2 = np.linalg.norm(vector1, ord=ord), np.linalg.norm(vector2, ord=ord)
                        # vector1_contracted = (2.0 - 1.0 / vector1.max()) / vector1.max() * vector1
                        # vector2_contracted = (2.0 - 1.0 / vector2.max()) / vector2.max() * vector2
                        vector1_contracted = (2.0 - 1.0 / norm1) / norm1 * vector1
                        vector2_contracted = (2.0 - 1.0 / norm2) / norm2 * vector2

                        mag_orig = vector1 > vector2
                        mag_cont = vector1_contracted > vector2_contracted

                        if (mag_orig != mag_cont).any():
                            print(
                                f"vector1: {np.array2string(vector1)}, vector1_contracted: {np.array2string(vector1_contracted)}"
                            )
                            print(
                                f"vector2: {np.array2string(vector2)}, vector2_contracted: {np.array2string(vector2_contracted)}"
                            )
                            sys.exit(0)
