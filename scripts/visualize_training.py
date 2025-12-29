"""Visualize training results from results.csv"""

import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Read CSV
epochs = []
map50 = []
map95 = []
precision = []
recall = []

with open(r'd:\Github\EV-PassengerDection-RL\results\coco_full\passenger_detection5\results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(float(row['epoch'])))
        map50.append(float(row['metrics/mAP50(B)']))
        map95.append(float(row['metrics/mAP50-95(B)']))
        precision.append(float(row['metrics/precision(B)']))
        recall.append(float(row['metrics/recall(B)']))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Results - COCO Person Detection (YOLOv11m)', fontsize=16, fontweight='bold')

# Plot 1: mAP curves
ax1 = axes[0, 0]
ax1.plot(epochs, map50, 'b-o', linewidth=2, label='mAP@50', markersize=4)
ax1.plot(epochs, map95, 'r-s', linewidth=2, label='mAP@50-95', markersize=4)
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('mAP', fontsize=10)
ax1.set_title('Mean Average Precision', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Precision & Recall
ax2 = axes[0, 1]
ax2.plot(epochs, precision, 'g-o', linewidth=2, label='Precision', markersize=4)
ax2.plot(epochs, recall, 'orange', marker='s', linewidth=2, label='Recall', markersize=4)
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Score', fontsize=10)
ax2.set_title('Precision & Recall', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: mAP@50 only
ax3 = axes[1, 0]
ax3.fill_between(epochs, 0, map50, alpha=0.3, color='blue')
ax3.plot(epochs, map50, 'b-o', linewidth=3, markersize=5)
ax3.set_xlabel('Epoch', fontsize=10)
ax3.set_ylabel('mAP@50', fontsize=10)
ax3.set_title('mAP@50 Progress', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
current_map = map50[-1]
ax3.text(len(epochs)*0.5, max(map50)*0.5, f'Current: {current_map:.3f}', 
         fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 4: All metrics normalized
ax4 = axes[1, 1]
ax4.plot(epochs, precision, 'g-', linewidth=2, label='Precision', marker='o', markersize=4)
ax4.plot(epochs, recall, 'orange', linewidth=2, label='Recall', marker='s', markersize=4)
ax4.plot(epochs, map50, 'b-', linewidth=2, label='mAP@50', marker='^', markersize=4)
ax4.plot(epochs, map95, 'r-', linewidth=2, label='mAP@50-95', marker='d', markersize=4)
ax4.set_xlabel('Epoch', fontsize=10)
ax4.set_ylabel('Score', fontsize=10)
ax4.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'd:\Github\EV-PassengerDection-RL\results\coco_full\passenger_detection5\training_summary.png', 
            dpi=150, bbox_inches='tight')
print("✅ Training summary visualization saved!")
print(f"📊 Latest metrics (Epoch {epochs[-1]}):")
print(f"   • mAP@50: {map50[-1]:.4f}")
print(f"   • mAP@50-95: {map95[-1]:.4f}")
print(f"   • Precision: {precision[-1]:.4f}")
print(f"   • Recall: {recall[-1]:.4f}")
