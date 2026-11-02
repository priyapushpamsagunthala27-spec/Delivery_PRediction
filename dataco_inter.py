import os
import json
import matplotlib.pyplot as plt

def generate_dataco_summary_slide(base_dir="outputs/local_dataco", output_img="dataco_intermediary_analysis.png"):
    clients = ["africa", "europe", "latam", "pacific_asia", "usca"]
    extracted_data = []

    # 1. Parse client metrics on the fly from their respective output directories
    for client in clients:
        status_path = os.path.join(base_dir, f"client_{client}", "live_status.json")
        
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            extracted_data.append({
                "client": client.upper().replace("_", " "),
                "accuracy": f"{data.get('val_accuracy', 0.0) * 100:.2f}%",
                "precision": f"{data.get('val_precision', 0.0) * 100:.2f}%",
                "recall": f"{data.get('val_recall', 0.0) * 100:.2f}%",
                "f1": f"{data.get('val_f1', 0.0) * 100:.2f}%",
                "best_epoch": f"Epoch {data.get('best_epoch', 0)}",
                "total_epochs": f"{data.get('current_epoch', 0)} epochs",
                "samples": f"{data.get('val_samples', 0) + data.get('train_samples', 0):,}"
            })
        else:
            # Fallback placeholder if directories haven't finished execution yet
            extracted_data.append({
                "client": client.upper().replace("_", " "),
                "accuracy": "N/A", "precision": "N/A", "recall": "N/A", "f1": "N/A",
                "best_epoch": "N/A", "total_epochs": "N/A", "samples": "N/A"
            })

    # 2. Set up a pristine high-DPI figure canvas sized perfectly for a PPT slide
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor('#F8F9FA')  # Premium crisp light background
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')

    # 3. Slide Headers
    plt.text(0.02, 0.90, "DATACO LOCAL MODELS: PERFORMANCE SUMMARY", 
             fontsize=16, fontweight='bold', color='#1A2530', transform=ax.transAxes)
    plt.text(0.02, 0.84, "Intermediary Baseline Evaluation Metrics Across Standalone Regional Clients", 
             fontsize=11, color='#5A6A85', transform=ax.transAxes)

    # 4. Define Table Grid Layout Coordinates
    columns = ["Regional Client", "Accuracy", "Precision", "Recall", "F1 Score", "Best Epoch", "Total Budget", "Total Dataset"]
    col_positions = [0.03, 0.18, 0.30, 0.42, 0.54, 0.67, 0.80, 0.91]
    y_start = 0.68

    # Table Top Frame border
    plt.axhline(y=y_start+0.04, xmin=0.02, xmax=0.98, color='#1A2530', linewidth=1.5)
    
    # Render Column Titles
    for col, pos in zip(columns, col_positions):
        plt.text(pos, y_start, col, fontsize=11, fontweight='bold', color='#1A2530', transform=ax.transAxes)
        
    # Table Header Separation Line
    plt.axhline(y=y_start-0.02, xmin=0.02, xmax=0.98, color='#A1A1AA', linewidth=1)

    # 5. Populate Data Rows
    y_offset = y_start - 0.09
    for i, row in enumerate(extracted_data):
        # Apply zebra striping row backgrounds to maximize structural legibility
        if i % 2 == 1:
            row_bg = plt.Rectangle((0.02, y_offset-0.02), 0.96, 0.065, facecolor='#F1F5F9', edgecolor='none', transform=ax.transAxes)
            ax.add_patch(row_bg)
            
        plt.text(col_positions[0], y_offset, row["client"], fontsize=10.5, fontweight='semibold', color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[1], y_offset, row["accuracy"], fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[2], y_offset, row["precision"], fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[3], y_offset, row["recall"], fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[4], y_offset, row["f1"], fontsize=10.5, color='#27272A', transform=ax.transAxes)
        plt.text(col_positions[5], y_offset, row["best_epoch"], fontsize=10.5, color='#52525B', transform=ax.transAxes)
        plt.text(col_positions[6], y_offset, row["total_epochs"], fontsize=10.5, color='#52525B', transform=ax.transAxes)
        plt.text(col_positions[7], y_offset, row["samples"], fontsize=10.5, color='#52525B', transform=ax.transAxes)
        
        y_offset -= 0.08

    # Table Base Frame Line
    plt.axhline(y=y_offset+0.06, xmin=0.02, xmax=0.98, color='#1A2530', linewidth=1.2)

    # Save out the image asset
    plt.savefig(output_img, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"DataCo summary table slide successfully generated at: {output_img}")

if __name__ == "__main__":
    generate_dataco_summary_slide()
