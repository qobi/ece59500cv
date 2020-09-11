# RANSAC
for model in models:
    for m1 in model:
        for m2 in model:
            for m3 in model:
                for m4 in model:
                    for i1 in image:
                        for i2 in image:
                            for i3 in image:
                                for i4 in image:
                                    camera_parameters, pose, articulation_parameters = inverse_project(m1, m2, m3, m4, i1, i2, i3, m4)
