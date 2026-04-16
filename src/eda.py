from pathlib import Path
import matplotlib
matplotlib.use('Agg') #since gui fails 
import matplotlib.pyplot as plt
train_folder = "/datasets/tdt4265/Poles2025/roadpoles_v1/train/labels"
validation_labels = "/datasets/tdt4265/Poles2025/roadpoles_v1/valid/labels"
w = []
h = []
train_boxes = 0

for f in Path(train_folder).glob("*.txt"):
    with open(f, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                w.append(float(parts[3]))
                h.append(float(parts[4]))
                train_boxes += 1

validation_boxes = 0
for f in Path(validation_labels).glob("*.txt"):
    with open(f, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                validation_boxes += 1

print("Train images:", 322)
print("Val images:", 92)
print("Train boxes:", train_boxes)
print("Val boxes:", validation_boxes)
print("avg poles/images:", train_boxes / 322)
print(f"width  — min: {min(w):.4f}, max: {max(w):.4f}, avg: {sum(w)/len(w):.4f}")
print(f"height — min: {min(h):.4f}, max: {max(h):.4f}, avg: {sum(h)/len(h):.4f}")

plt.figure()
plt.hist(w, bins=30)
plt.title("train widths")
plt.xlabel("width")
plt.ylabel("count")
Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/train_widths.png")
plt.figure()
plt.hist(h, bins=30)
plt.title("train heights")
plt.xlabel("height")
plt.ylabel("count")
plt.savefig("outputs/train_heights.png")

