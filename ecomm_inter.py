import os
import json
import matplotlib.pyplot as plt

def generate_ppt_summary_slide(json_path="outputs/local_ecommerce/final_summary.json", output_img="ecommerce_intermediary_analysis.png"):
    # 1. Load the summary data
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not locate the summary file at {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # 2. Set up a clean, high-DPI figure canvas mimicking widescreen PPT formats
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor('#F8F9FA')  # Light slate background
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')

    # 3. Slide Headers
    plt.text(0.02, 0.90, "E-COMMERCE LOCAL MODELS: PERFORMANCE SUMMARY", 
             fontsize=16, fontweight='bold', color='#1A2530', transform=ax.transAxes)
    plt.text(0.02, 0.84, "Intermediary Baseline Evaluation Metrics Across All Standalone Clients", 
             fontsize=11, color='#5A6A85', transform=ax.transAxes)

    # 4. Table Layout Configuration
    columns = ["Client ID", "Accuracy", "Precision", "Recall", "F1 Score", "Best Epoch", "Total Epochs", "Samples"]
    col_positions = [0.03, 0.16, 0.28, 0.40, 0.52, 0.66, 0.79, 0.90]
    y_start = 0.68

    # Table Top Border Line
    plt.axhline(y=y_start+0.04, xmin=0.02, xmax=0.98, color='#1A2530', linewidth=1.5)
    
    # Render Column Headers
    for col, pos in zip(columns, col_positions):
        plt.text(pos, y_start, col, fontsize=11, fontweight='bold', color='#1A2530', transform=ax.transAxes)
        
    # Header Underline
    plt.axhline(y=y_start-0.02, xmin=0.02, xmax=0.98, color='#A1A1AA', linewidth=1)

    # 5. Populate Rows
    y_offset = y_start - 0.09
    for i, client in enumerate(summary_data['clients']):
        # Alternating row background highlights for scannability
        if i % 2 == 1:
            row_bg = plt.Rectangle((0.02, y_offset-0.02), 0.96, 0.065, facecolor='#F1F5F9', edgecolor='none', transform=ax.transAxes)
            ax.add_patch(row_bg)
            
        plt.text(col_positions[0], y_offset, f"Client {client['client']}", fontsize=10.5, fontweight='semibold', color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[1], y_offset, f"{client['final_accuracy']*100:.2f}%", fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[2], y_offset, f"{client['final_precision']*100:.2f}%", fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[3], y_offset, f"{client['final_recall']*100:.2f}%", fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[4], y_offset, f"{client['final_f1']*100:.2f}%", fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[5], y_offset, f"Epoch {client['best_epoch']}", fontsize=10.5, color='#52525B', transform=ax.transAxes)
        plt.text(col_positions[6], y_offset, f"{client['total_epochs_run']} epochs", fontsize=10.5, color='#52525B', transform=ax.transAxes)
        plt.text(col_positions[7], y_offset, f"{client['train_samples']:,}", fontsize=10.5, color='#52525B', transform=ax.transAxes)
        
        y_offset -= 0.08

    # Table Bottom Border Line
    plt.axhline(y=y_offset+0.06, xmin=0.02, xmax=0.98, color='#1A2530', linewidth=1.2)

    # Save out the image asset
    plt.savefig(output_img, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"Summary table slide successfully generated at: {output_img}")

if __name__ == "__main__":
    generate_ppt_summary_slide(json_path="outputs/local_ecommerce/final_summary.json")
