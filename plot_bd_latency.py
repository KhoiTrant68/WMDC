import matplotlib.pyplot as plt

# Data
data = [
    ("Cheng20-Parallel", 40, 4, 24.52, '#c1e7c1'),
    ("Minnen20", 100, 1, 41.77, '#8eeded'),
    ("ELIC", 125, -7, 33.29, '#78a2a2'),
    ("TCM-large", 145, -12, 75.90, '#a0cfff'),
    ("LALIC (Ours)", 155, -15.5, 63.24, 'red'),
    ("MambaVC", 225, -10, 47.88, '#70a470'),
    ("MLIC++", 270, -15, 83.27, '#c3a0cc'),
    ("FAT", 380, -14.5, 69.78, '#dede4e')
]

fig, ax = plt.subplots(figsize=(8, 6))

# 1. Grid Logic (Add manual vertical line for the slash position)
ax.grid(True, which='major', linestyle='-', linewidth=0.6, color='0.8', zorder=1)
break_x_coord = 320 # This is where the slash sits
ax.axvline(break_x_coord, color='0.8', linewidth=0.6, zorder=1) 

# 2. Plot the bubbles
for name, x, y, size, color in data:
    scale = size * 18
    ax.scatter(x, y, s=scale, color=color, alpha=0.9, edgecolors='none', zorder=4)
    
    label_color = 'red' if 'Ours' in name else 'black'
    ax.text(x, y - 2.2, f"{name}\n{size}M", ha='center', va='top', 
            fontsize=9, color=label_color, fontweight='bold' if 'Ours' in name else 'normal')

# 3. Reference line
ax.axhline(0, color='brown', linestyle='--', linewidth=1.2, zorder=2)
ax.text(380, 0.8, "VTM-9.1", color='brown', ha='center', fontsize=10)

# 4. Slashes on the frame edges
slash_pos = 0.745 # Relative position (0 to 1)
slash_style = dict(transform=ax.transAxes, color='black', clip_on=False, fontsize=14)

# We use two slashes close together to match the '//' look
for offset in [-0.005, 0.005]:
    ax.text(slash_pos + offset, 0, '/', ha='center', va='center', **slash_style)
    ax.text(slash_pos + offset, 1, '/', ha='center', va='center', **slash_style)

# 5. Ticks & Labels
ticks = [20, 50, 100, 150, 200, 250, 370, 410]
tick_labels = ["20", "50", "100", "150", "200", "250", "4.1×10⁵", "4.3×10⁵"]
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)

# 6. Final Styling
ax.set_xlim(10, 430)
ax.set_ylim(7, -18)
ax.set_xlabel("Decoding Latency (ms)", fontsize=11)
ax.set_ylabel("BD-Rate over VTM-9.1 (%)", fontsize=11)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_visible(True)

plt.tight_layout()
plt.show()