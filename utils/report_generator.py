import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors


def generate_report(
    output_pdf,
    peptide_sequence,
    drug_metrics,
    docking_score=None,
    gnn_score=None,
    snapshot_path=None
):

    styles = getSampleStyleSheet()
    elements = []

    
    # TITLE
    
    elements.append(Paragraph("AI Peptide Drug Discovery Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    
    # SEQUENCE
    
    elements.append(Paragraph(f"<b>Peptide Sequence:</b> {peptide_sequence}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    
    # SAFE METRICS EXTRACTION
    
    charge = drug_metrics.get("charge", "NA")
    length = drug_metrics.get("length", "NA")
    pI = drug_metrics.get("pI", "NA")
    mw = drug_metrics.get("mw", "NA")
    aromatic = drug_metrics.get("aromatic", "NA")
    hydro = drug_metrics.get("hydrophobicity", "NA")

    
    # METRICS TABLE
    
    data = [
        ["Metric", "Value"],
        ["Length", length],
        ["Molecular Weight", mw],
        ["Charge", charge],
        ["pI", pI],
        ["Aromaticity", aromatic],
        ["Hydrophobicity", hydro],
    ]

    table = Table(data, colWidths=[2.5 * inch, 2.5 * inch])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    
    # SCORES
    
    elements.append(Paragraph("<b>Scoring Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    if docking_score is not None:
        elements.append(Paragraph(f"HADDOCK Score: {docking_score}", styles["Normal"]))

    if gnn_score is not None:
        elements.append(Paragraph(f"Deep Learning Affinity Score: {gnn_score}", styles["Normal"]))

    elements.append(Spacer(1, 20))

    
    # RULES / DRUG-LIKE STATUS
    
    rules = drug_metrics.get("rules", {})

    if rules:
        elements.append(Paragraph("<b>Drug-Likeness Rules</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        for k, v in rules.items():
            status = "PASS" if v else "FAIL"
            elements.append(Paragraph(f"{k}: {status}", styles["Normal"]))

        elements.append(Spacer(1, 20))

    
    # DOCKING IMAGE
    
    if snapshot_path and os.path.exists(snapshot_path):

        elements.append(Paragraph("<b>Docking Visualization</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        img = Image(snapshot_path, width=5 * inch, height=4 * inch)
        elements.append(img)

    
    # BUILD PDF
    
    doc = SimpleDocTemplate(output_pdf)
    doc.build(elements)

    return output_pdf
