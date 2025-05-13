import matplotlib.pyplot as plt
import numpy as np

# ROM and diversity values
rom_labels = ["Full Body", "Left Hand", "Right Hand", "Lower Body"]
rom_values = [0.85, 0.22, 0.20, 0.40]

div_labels = ["Full Body", "Left Hand", "Right Hand", "Lower Body"]
div_values = [1.0, 0.35, 0.33, 0.49]

# Plot ROM heatmap
plt.figure(figsize=(6, 2))
plt.imshow([rom_values], cmap='YlOrRd', aspect='auto')
plt.xticks(ticks=np.arange(len(rom_labels)), labels=rom_labels, rotation=45, ha='right')
plt.yticks([0], ["ROM (m)"])
for i, val in enumerate(rom_values):
    plt.text(i, 0, f"{val:.2f}", ha='center', va='center', color='black')
plt.title("Range of Motion (ROM) Heatmap")
plt.colorbar(label='Meters')
plt.tight_layout()
plt.savefig("rom_heatmap.png", dpi=300)
plt.close()

# Plot Diversity heatmap
plt.figure(figsize=(6, 2))
plt.imshow([div_values], cmap='PuBu', aspect='auto')
plt.xticks(ticks=np.arange(len(div_labels)), labels=div_labels, rotation=45, ha='right')
plt.yticks([0], ["Pose Diversity"])
for i, val in enumerate(div_values):
    plt.text(i, 0, f"{val:.2f}", ha='center', va='center', color='black')
plt.title("Pose Diversity Heatmap")
plt.colorbar(label='Diversity Score')
plt.tight_layout()
plt.savefig("pose_diversity_heatmap.png", dpi=300)
plt.close()
