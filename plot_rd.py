import bjontegaard as bd
import matplotlib.pyplot as plt

# =========================================================
# 1. Define your data (BPP and PSNR for each method)
# =========================================================
# Format: "Method Name": ([bpp_points], [psnr_points], "marker", "color")

methods_data = {
    # Anchor Method (Usually VVC or a standard baseline)
    "VVC (VTM)": ([0.160, 0.260, 0.370, 0.520, 0.730, 0.980],[29.30, 31.00, 32.80, 34.80, 36.90, 38.80], 
        'o', '#000000' # Black circle
    ),
    
    # Traditional Codecs
    "HEVC (HM)": ([0.180, 0.290, 0.410, 0.580, 0.810, 1.050],[28.50, 30.20, 32.00, 34.00, 36.10, 37.90], 
        'x', '#808080' # Gray X
    ),
    "BPG": ([0.190, 0.300, 0.430, 0.600, 0.840, 1.100],[28.20, 29.90, 31.70, 33.70, 35.80, 37.50], 
        '+', '#A9A9A9' # Light Gray +
    ),

    # Learned Baselines (Mock data representing typical literature values)
    "Minnen2018": ([0.170, 0.280, 0.400, 0.560, 0.790, 1.020],[28.90, 30.60, 32.40, 34.30, 36.30, 38.20], 
        '^', '#1f77b4' # Blue triangle
    ),
    "Cheng2020": ([0.155, 0.255, 0.365, 0.510, 0.720, 0.970],[29.40, 31.10, 32.90, 34.90, 37.00, 38.90], 
        'v', '#ff7f0e' # Orange down-triangle
    ),
    "TCM (2023)": ([0.145, 0.245, 0.345, 0.490, 0.690, 0.940],[29.60, 31.30, 33.20, 35.10, 37.30, 39.20], 
        's', '#2ca02c' # Green square
    ),

    # Your Proposed Method
    "WMDC (Ours)": ([0.135, 0.230, 0.330, 0.470, 0.660, 0.900],[29.80, 31.60, 33.50, 35.50, 37.70, 39.60], 
        '*', '#d62728' # Red star
    )
}

# =========================================================
# 2. Setup Plotting
# =========================================================
plt.figure(figsize=(10, 7))
anchor_name = "VVC (VTM)"
anchor_bpp, anchor_psnr, _, _ = methods_data[anchor_name]

print(f"--- BD-Rate compared to {anchor_name} ---")

# =========================================================
# 3. Loop through methods, calculate BD-Rate, and Plot
# =========================================================
for name, (bpp, psnr, marker, color) in methods_data.items():
    
    # Calculate BD-Rate vs Anchor
    if name == anchor_name:
        label = f"{name} (Anchor)"
    else:
        try:
            # Using 'akima' method for standard stability
            bd_rate = bd.bd_rate(anchor_bpp, anchor_psnr, bpp, psnr, 'akima')
            label = f"{name} (BD-Rate: {bd_rate:+.2f}%)"
            print(f"{name: <15}: {bd_rate:+.2f}%")
        except Exception as e:
            # Fallback if curves don't overlap enough for interpolation
            label = f"{name}"
            print(f"{name: <15}: N/A (Curve overlap issue)")

    # Plot settings: Make "Ours" stand out
    linewidth = 2.5 if "Ours" in name else 1.5
    markersize = 10 if "Ours" in name else 7
    zorder = 10 if "Ours" in name else 5  # Bring ours to the front

    plt.plot(bpp, psnr, marker=marker, color=color, linestyle='-', 
             linewidth=linewidth, markersize=markersize, label=label, zorder=zorder)

# =========================================================
# 4. Formatting for Academic Paper
# =========================================================
plt.title('Rate-Distortion Performance on Kodak Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Bits per Pixel (BPP)', fontsize=14)
plt.ylabel('PSNR (dB)', fontsize=14)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Position legend in bottom right (usually best for RD curves)
plt.legend(loc='lower right', fontsize=11, framealpha=0.9, edgecolor='black')

# Save as PDF (Best for LaTeX) and PNG
plt.tight_layout()
plt.savefig("rd_curve_kodak.pdf", format='pdf', dpi=300)
plt.savefig("rd_curve_kodak.png", format='png', dpi=300)

print("\nSaved plots as 'rd_curve_kodak.pdf' and 'rd_curve_kodak.png'")
plt.show()