# Computer Vision and Applications — Coursework

Assignments for **Computer Vision and Applications (CI5336701)** at **NTUST**.

---

## Assignments

### [Homework 1 — Drawing a 3D Trajectory onto Camera Images](Homework_1/README.md)

Given the 3D flight path of a football (384 points in world coordinates) and two photographs of the same virtual stadium taken by cameras with known parameters, compute where every 3D point lands in each photo and draw the trajectory onto the images.

This is the **pinhole camera model** put to work — `x = K[R|T]X` — covering homogeneous coordinates, the intrinsic matrix `K` (the lens), the extrinsic matrix `[R|T]` (the camera's position and orientation), and the perspective divide that actually creates the sense of depth. The two cameras are deliberately very different: one stands at eye level beside the pitch with a 39.6° field of view, the other looks down from 27 m up through a narrower 25.4° telephoto lens. The same 3D path therefore appears as a long diagonal in one image and a tight hook in the other.

The result is verified visually: the first trajectory point lands exactly on the ball at the player's foot in **both** images, which two independent viewpoints would never agree on if the maths were wrong.

📄 **[Read the full write-up →](Homework_1/README.md)**

| Camera 1 (pitch-side) | Camera 2 (aerial) |
|---|---|
| ![Trajectory on camera 1](Homework_1/F11115117_1.jpg) | ![Trajectory on camera 2](Homework_1/F11115117_2.jpg) |

---

### [Homework 2 — Swapping Two Photo Frames with a Homography](Homework_2/README.md)

Given a single photo of an art gallery holding two picture frames — one large, close and turned away from the camera, the other small, distant and nearly frontal — estimate the projective transform between them from four hand-picked corner pairs and swap the paintings, so each one hangs in the other's frame with the correct new perspective.

Where Homework 1 went from 3D to 2D, this one stays in the image plane and exploits a different fact: **both paintings are flat**. Any two views of the same plane are related by a single 3×3 **homography** — the most general transform that still maps straight lines to straight lines, with **8 degrees of freedom** rather than 9, because homogeneous coordinates are only defined up to scale. The write-up covers the transform hierarchy (translation → Euclidean → similarity → affine → homography), the **Direct Linear Transform** that turns the non-linear mapping into a solvable 8×8 linear system, why four point pairs are exactly enough, and the bitwise-mask compositing trick that drops a warped image into a hand-defined region.

Because a homography is invertible, the same matrix does both halves of the swap: `H` sends the left painting right, and `H⁻¹` sends the right painting left.

📄 **[Read the full write-up →](Homework_2/README.md)**

| Original | Paintings swapped |
|---|---|
| ![Original gallery](Homework_2/Swap_ArtGallery.jpg) | ![Paintings swapped](Homework_2/F11115117.jpg) |

---

## Tooling

Python, in Jupyter notebooks, using **NumPy** (matrix maths), **OpenCV** (`opencv-python`, for image I/O and drawing) and **Matplotlib** (inline display).

```bash
pip install numpy opencv-python matplotlib
```
