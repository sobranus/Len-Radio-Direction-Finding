import tkinter as tk
import math

root = tk.Tk()
root.title("Bearing-Only Triangulation + Distance Display")

WIDTH = 800
HEIGHT = 600
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# Detector positions
det1 = (200, HEIGHT - 50)
det2 = (600, HEIGHT - 50)

canvas.create_oval(det1[0]-10, det1[1]-10, det1[0]+10, det1[1]+10, fill="blue")
canvas.create_oval(det2[0]-10, det2[1]-10, det2[0]+10, det2[1]+10, fill="red")

# Drawing placeholders
true_obj = canvas.create_oval(0, 0, 0, 0)
est_obj = None

ray1 = None
ray2 = None
line1 = None
line2 = None
text1 = None
text2 = None


def triangulate(p1, angle1, p2, angle2):
    """Return intersection point of two infinite rays."""
    x1, y1 = p1
    x2, y2 = p2

    dx1, dy1 = math.cos(angle1), math.sin(angle1)
    dx2, dy2 = math.cos(angle2), math.sin(angle2)

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-6:
        return None  # parallel

    t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return x1 + t1 * dx1, y1 + t1 * dy1


def update(event):
    global true_obj, est_obj, ray1, ray2, line1, line2, text1, text2

    ox, oy = event.x, event.y  # true object position

    # Draw true object
    canvas.coords(true_obj, ox-8, oy-8, ox+8, oy+8)
    canvas.itemconfig(true_obj, fill="green")

    # Clear previous drawings
    for item in [ray1, ray2, est_obj, line1, line2, text1, text2]:
        if item:
            canvas.delete(item)

    # Bearin𝘨 measurements
    angle1 = math.atan2(oy - det1[1], ox - det1[0])
    angle2 = math.atan2(oy - det2[1], ox - det2[0])

    ray_len = 2000

    # Draw bearing rays
    ray1 = canvas.create_line(det1[0], det1[1],
                              det1[0] + math.cos(angle1) * ray_len,
                              det1[1] + math.sin(angle1) * ray_len,
                              fill="blue")

    ray2 = canvas.create_line(det2[0], det2[1],
                              det2[0] + math.cos(angle2) * ray_len,
                              det2[1] + math.sin(angle2) * ray_len,
                              fill="red")

    # Triangulate
    result = triangulate(det1, angle1, det2, angle2)

    if result:
        ix, iy = result
        est_obj = canvas.create_oval(ix-6, iy-6, ix+6, iy+6, fill="yellow")

    # True distances
    d1 = math.hypot(ox - det1[0], oy - det1[1])
    d2 = math.hypot(ox - det2[0], oy - det2[1])

    # Draw lines from detectors to TRUE object
    line1 = canvas.create_line(det1[0], det1[1], ox, oy, fill="cyan", width=2)
    line2 = canvas.create_line(det2[0], det2[1], ox, oy, fill="magenta", width=2)

    # Midpoints for distance labels
    mid1 = ((det1[0] + ox) / 2, (det1[1] + oy) / 2)
    mid2 = ((det2[0] + ox) / 2, (det2[1] + oy) / 2)

    text1 = canvas.create_text(mid1[0], mid1[1] - 10,
                               text=f"{d1:.2f}", fill="blue", font=("Arial", 12))

    text2 = canvas.create_text(mid2[0], mid2[1] - 10,
                               text=f"{d2:.2f}", fill="red", font=("Arial", 12))


canvas.bind("<Button-1>", update)
root.mainloop()
