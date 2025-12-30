"""
plan_formatter.py — Compliance Plan Output Formatters

Converts CompliancePlan objects into various output formats:
- Markdown (for chat display)
- JSON (for API/storage)
- PDF (for download/reports)
- HTML (for web display)

Author: Finvij Team
Phase: 3 (Days 19-21)
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

log = logging.getLogger("plan_formatter")

# Import readability analyzer
try:
    from readability_analyzer import analyze_readability
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    log.warning("Readability analyzer not available")


# ================================================================================
# MARKDOWN FORMATTER
# ================================================================================

class MarkdownFormatter:
    """Format compliance plan as Markdown"""

    @staticmethod
    def format(plan: Dict) -> str:
        """
        Format plan as Markdown for chat display.

        Args:
            plan: Plan dictionary (from CompliancePlan.to_dict())

        Returns:
            Markdown formatted string
        """
        md_lines = []

        # Header
        md_lines.append(f"# 📋 Compliance Plan v{plan.get('version', 1)}")
        md_lines.append("")
        md_lines.append(f"**Generated:** {plan.get('generated_at', 'Unknown')}")
        md_lines.append(f"**Entity Type:** {plan.get('entity_type', 'Unknown')}")
        md_lines.append(f"**Products:** {', '.join(plan.get('products', []))}")
        md_lines.append(f"**Requirements Covered:** {plan.get('requirements_count', 0)}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Summary
        md_lines.append("## Executive Summary")
        md_lines.append("")
        md_lines.append(plan.get("summary", "No summary available."))
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Priority Areas
        md_lines.append("## 🎯 Priority Compliance Areas")
        md_lines.append("")

        priority_areas = plan.get("priority_areas", [])
        if priority_areas:
            for i, area in enumerate(priority_areas, 1):
                criticality = area.get("criticality", "medium")
                emoji = "🔴" if criticality == "high" else "🟡" if criticality == "medium" else "🟢"

                md_lines.append(f"### {i}. {area.get('area', 'Unknown')} {emoji}")
                md_lines.append(f"**Criticality:** {criticality.upper()}")
                md_lines.append(f"**Reason:** {area.get('reason', 'N/A')}")
                md_lines.append("")
        else:
            md_lines.append("_No priority areas identified._")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # Timeline-based Actions
        md_lines.append("## 📅 Action Plan by Timeline")
        md_lines.append("")

        timeline_actions = plan.get("timeline_based_actions", {})

        # Immediate
        md_lines.append("### 🚨 IMMEDIATE (0-15 Days)")
        md_lines.append("")
        immediate = timeline_actions.get("immediate", [])
        if immediate:
            for i, action in enumerate(immediate, 1):
                md_lines.append(f"**{i}. {action.get('requirement', 'Unknown')}**")
                md_lines.append(f"- **Action:** {action.get('action', 'N/A')}")
                md_lines.append(f"- **Responsible:** {action.get('responsible', 'N/A')}")
                md_lines.append(f"- **Deliverable:** {action.get('deliverable', 'N/A')}")
                md_lines.append(f"- **Reference:** {action.get('rbi_reference', 'N/A')}")
                md_lines.append("")
        else:
            md_lines.append("_No immediate actions required._")
            md_lines.append("")

        # 30 Days
        md_lines.append("### ⏰ 30 DAYS")
        md_lines.append("")
        thirty_days = timeline_actions.get("30_days", [])
        if thirty_days:
            for i, action in enumerate(thirty_days, 1):
                md_lines.append(f"**{i}. {action.get('requirement', 'Unknown')}**")
                md_lines.append(f"- **Action:** {action.get('action', 'N/A')}")
                md_lines.append(f"- **Responsible:** {action.get('responsible', 'N/A')}")
                md_lines.append(f"- **Deliverable:** {action.get('deliverable', 'N/A')}")
                md_lines.append("")
        else:
            md_lines.append("_No 30-day actions._")
            md_lines.append("")

        # 90 Days
        md_lines.append("### 📆 90 DAYS")
        md_lines.append("")
        ninety_days = timeline_actions.get("90_days", [])
        if ninety_days:
            for i, action in enumerate(ninety_days, 1):
                md_lines.append(f"**{i}. {action.get('requirement', 'Unknown')}**")
                md_lines.append(f"- **Action:** {action.get('action', 'N/A')}")
                md_lines.append(f"- **Responsible:** {action.get('responsible', 'N/A')}")
                md_lines.append("")
        else:
            md_lines.append("_No 90-day actions._")
            md_lines.append("")

        # Ongoing
        md_lines.append("### 🔄 ONGOING")
        md_lines.append("")
        ongoing = timeline_actions.get("ongoing", [])
        if ongoing:
            for i, action in enumerate(ongoing, 1):
                md_lines.append(f"{i}. {action.get('requirement', 'Unknown')} - {action.get('action', 'N/A')}")
        else:
            md_lines.append("_No ongoing actions._")
        md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # Risk Areas
        md_lines.append("## ⚠️ Risk Areas")
        md_lines.append("")

        risk_areas = plan.get("risk_areas", [])
        if risk_areas:
            for i, risk in enumerate(risk_areas, 1):
                md_lines.append(f"### {i}. {risk.get('risk', 'Unknown')}")
                md_lines.append(f"**Impact:** {risk.get('impact', 'N/A')}")
                md_lines.append(f"**Mitigation:** {risk.get('mitigation', 'N/A')}")
                md_lines.append("")
        else:
            md_lines.append("_No risk areas identified._")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # Implementation Checklist
        md_lines.append("## ✅ Implementation Checklist")
        md_lines.append("")

        checklist = plan.get("implementation_checklist", {})
        if checklist:
            for area, steps in checklist.items():
                md_lines.append(f"### {area}")
                for step in steps:
                    md_lines.append(f"- [ ] {step}")
                md_lines.append("")
        else:
            md_lines.append("_No checklist available._")
            md_lines.append("")

        # Readability Analysis (if available)
        if READABILITY_AVAILABLE:
            try:
                full_text = "\n".join(md_lines)
                readability = analyze_readability(full_text)

                md_lines.append("---")
                md_lines.append("")
                md_lines.append("## 📊 Readability Analysis")
                md_lines.append("")
                md_lines.append(f"- **Reading Ease:** {readability.flesch_reading_ease:.1f} ({readability.readability_level})")
                md_lines.append(f"- **Grade Level:** {readability.flesch_kincaid_grade:.1f} ({readability.grade_level_interpretation})")
                md_lines.append(f"- **Avg Sentence Length:** {readability.avg_sentence_length:.1f} words")
                md_lines.append(f"- **Complex Words:** {readability.complex_word_percentage:.1f}%")
                md_lines.append("")
            except Exception as e:
                log.warning(f"Failed to add readability analysis: {e}")

        # Footer
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("_This plan was generated based on your entity profile and RBI regulatory requirements._")
        md_lines.append("_Use `/refine-plan` to request changes or additions._")

        return "\n".join(md_lines)


# ================================================================================
# JSON FORMATTER
# ================================================================================

class JSONFormatter:
    """Format compliance plan as JSON"""

    @staticmethod
    def format(plan: Dict, pretty: bool = True) -> str:
        """
        Format plan as JSON.

        Args:
            plan: Plan dictionary
            pretty: Whether to pretty-print

        Returns:
            JSON string
        """
        if pretty:
            return json.dumps(plan, indent=2, ensure_ascii=False)
        else:
            return json.dumps(plan, ensure_ascii=False)


# ================================================================================
# PDF FORMATTER
# ================================================================================

class PDFFormatter:
    """Format compliance plan as PDF"""

    @staticmethod
    def format(plan: Dict, output_path: str) -> bool:
        """
        Format plan as PDF and save to file.

        Args:
            plan: Plan dictionary
            output_path: Path to save PDF

        Returns:
            True if successful, False otherwise
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
            from reportlab.lib import colors

            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=30,
                alignment=TA_CENTER
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=12,
                spaceBefore=12
            )

            # Title
            story.append(Paragraph(f"Compliance Plan v{plan.get('version', 1)}", title_style))
            story.append(Spacer(1, 0.2 * inch))

            # Metadata table
            metadata = [
                ["Generated:", plan.get('generated_at', 'Unknown')],
                ["Entity Type:", plan.get('entity_type', 'Unknown')],
                ["Products:", ', '.join(plan.get('products', []))],
                ["Requirements Covered:", str(plan.get('requirements_count', 0))]
            ]

            metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))

            story.append(metadata_table)
            story.append(Spacer(1, 0.3 * inch))

            # Summary
            story.append(Paragraph("Executive Summary", heading_style))
            story.append(Paragraph(plan.get('summary', 'No summary available.'), styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))

            # Priority Areas
            story.append(Paragraph("Priority Compliance Areas", heading_style))
            for i, area in enumerate(plan.get('priority_areas', []), 1):
                story.append(Paragraph(
                    f"<b>{i}. {area.get('area', 'Unknown')}</b> (Criticality: {area.get('criticality', 'medium').upper()})",
                    styles['BodyText']
                ))
                story.append(Paragraph(f"<i>{area.get('reason', 'N/A')}</i>", styles['BodyText']))
                story.append(Spacer(1, 0.1 * inch))

            story.append(PageBreak())

            # Action Plan by Timeline
            story.append(Paragraph("Action Plan by Timeline", heading_style))

            timeline_actions = plan.get("timeline_based_actions", {})

            for timeline_name, timeline_label in [
                ("immediate", "IMMEDIATE (0-15 Days)"),
                ("30_days", "30 DAYS"),
                ("90_days", "90 DAYS"),
                ("ongoing", "ONGOING")
            ]:
                actions = timeline_actions.get(timeline_name, [])
                if actions:
                    story.append(Paragraph(f"<b>{timeline_label}</b>", styles['Heading3']))

                    for i, action in enumerate(actions, 1):
                        story.append(Paragraph(f"<b>{i}. {action.get('requirement', 'Unknown')}</b>", styles['BodyText']))
                        story.append(Paragraph(f"Action: {action.get('action', 'N/A')}", styles['BodyText']))
                        story.append(Paragraph(f"Responsible: {action.get('responsible', 'N/A')}", styles['BodyText']))
                        story.append(Paragraph(f"Deliverable: {action.get('deliverable', 'N/A')}", styles['BodyText']))

                        if action.get('rbi_reference'):
                            story.append(Paragraph(f"Reference: {action.get('rbi_reference')}", styles['BodyText']))

                        story.append(Spacer(1, 0.1 * inch))

                    story.append(Spacer(1, 0.2 * inch))

            # Risk Areas
            story.append(PageBreak())
            story.append(Paragraph("Risk Areas", heading_style))

            for i, risk in enumerate(plan.get('risk_areas', []), 1):
                story.append(Paragraph(f"<b>{i}. {risk.get('risk', 'Unknown')}</b>", styles['BodyText']))
                story.append(Paragraph(f"Impact: {risk.get('impact', 'N/A')}", styles['BodyText']))
                story.append(Paragraph(f"Mitigation: {risk.get('mitigation', 'N/A')}", styles['BodyText']))
                story.append(Spacer(1, 0.15 * inch))

            # Implementation Checklist
            story.append(Paragraph("Implementation Checklist", heading_style))

            checklist = plan.get("implementation_checklist", {})
            for area, steps in checklist.items():
                story.append(Paragraph(f"<b>{area}</b>", styles['BodyText']))
                for step in steps:
                    story.append(Paragraph(f"☐ {step}", styles['BodyText']))
                story.append(Spacer(1, 0.1 * inch))

            # Build PDF
            doc.build(story)
            log.info(f"PDF generated successfully: {output_path}")
            return True

        except ImportError:
            log.error("reportlab not installed. Install with: pip install reportlab")
            return False
        except Exception as e:
            log.error(f"Failed to generate PDF: {e}")
            return False


# ================================================================================
# HTML FORMATTER
# ================================================================================

class HTMLFormatter:
    """Format compliance plan as HTML"""

    @staticmethod
    def format(plan: Dict) -> str:
        """
        Format plan as HTML.

        Args:
            plan: Plan dictionary

        Returns:
            HTML string
        """
        html_parts = []

        # Header
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance Plan v{version}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f4788;
            border-bottom: 3px solid #1f4788;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c5aa0;
            margin-top: 30px;
        }}
        .metadata {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .priority-area {{
            background-color: #fff9e6;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
        }}
        .action-item {{
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .risk-area {{
            background-color: #ffebee;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #f44336;
            border-radius: 4px;
        }}
        .timeline-immediate {{ border-left-color: #f44336; }}
        .timeline-30days {{ border-left-color: #ff9800; }}
        .timeline-90days {{ border-left-color: #2196F3; }}
        .timeline-ongoing {{ border-left-color: #4CAF50; }}
        ul {{ padding-left: 20px; }}
        .checklist {{ list-style-type: none; }}
        .checklist li::before {{ content: "☐ "; margin-right: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📋 Compliance Plan v{version}</h1>
        """.format(version=plan.get('version', 1)))

        # Metadata
        html_parts.append(f"""
        <div class="metadata">
            <p><strong>Generated:</strong> {plan.get('generated_at', 'Unknown')}</p>
            <p><strong>Entity Type:</strong> {plan.get('entity_type', 'Unknown')}</p>
            <p><strong>Products:</strong> {', '.join(plan.get('products', []))}</p>
            <p><strong>Requirements Covered:</strong> {plan.get('requirements_count', 0)}</p>
        </div>
        """)

        # Summary
        html_parts.append(f"""
        <h2>Executive Summary</h2>
        <p>{plan.get('summary', 'No summary available.')}</p>
        """)

        # Priority Areas
        html_parts.append("<h2>🎯 Priority Compliance Areas</h2>")
        for i, area in enumerate(plan.get('priority_areas', []), 1):
            html_parts.append(f"""
            <div class="priority-area">
                <h3>{i}. {area.get('area', 'Unknown')}</h3>
                <p><strong>Criticality:</strong> {area.get('criticality', 'medium').upper()}</p>
                <p><strong>Reason:</strong> {area.get('reason', 'N/A')}</p>
            </div>
            """)

        # Timeline Actions
        html_parts.append("<h2>📅 Action Plan by Timeline</h2>")

        timeline_actions = plan.get("timeline_based_actions", {})

        for timeline_key, timeline_label, css_class in [
            ("immediate", "IMMEDIATE (0-15 Days)", "timeline-immediate"),
            ("30_days", "30 DAYS", "timeline-30days"),
            ("90_days", "90 DAYS", "timeline-90days"),
            ("ongoing", "ONGOING", "timeline-ongoing")
        ]:
            actions = timeline_actions.get(timeline_key, [])
            if actions:
                html_parts.append(f"<h3>{timeline_label}</h3>")
                for i, action in enumerate(actions, 1):
                    html_parts.append(f"""
                    <div class="action-item {css_class}">
                        <h4>{i}. {action.get('requirement', 'Unknown')}</h4>
                        <p><strong>Action:</strong> {action.get('action', 'N/A')}</p>
                        <p><strong>Responsible:</strong> {action.get('responsible', 'N/A')}</p>
                        <p><strong>Deliverable:</strong> {action.get('deliverable', 'N/A')}</p>
                        <p><strong>Reference:</strong> {action.get('rbi_reference', 'N/A')}</p>
                    </div>
                    """)

        # Risk Areas
        html_parts.append("<h2>⚠️ Risk Areas</h2>")
        for i, risk in enumerate(plan.get('risk_areas', []), 1):
            html_parts.append(f"""
            <div class="risk-area">
                <h3>{i}. {risk.get('risk', 'Unknown')}</h3>
                <p><strong>Impact:</strong> {risk.get('impact', 'N/A')}</p>
                <p><strong>Mitigation:</strong> {risk.get('mitigation', 'N/A')}</p>
            </div>
            """)

        # Checklist
        html_parts.append("<h2>✅ Implementation Checklist</h2>")
        checklist = plan.get("implementation_checklist", {})
        for area, steps in checklist.items():
            html_parts.append(f"<h3>{area}</h3>")
            html_parts.append('<ul class="checklist">')
            for step in steps:
                html_parts.append(f"<li>{step}</li>")
            html_parts.append("</ul>")

        # Footer
        html_parts.append("""
    </div>
</body>
</html>
        """)

        return "".join(html_parts)


# ================================================================================
# PLAN FORMATTER (Main Interface)
# ================================================================================

class PlanFormatter:
    """Main formatter interface"""

    @staticmethod
    def to_markdown(plan: Dict) -> str:
        """Format plan as Markdown"""
        return MarkdownFormatter.format(plan)

    @staticmethod
    def to_json(plan: Dict, pretty: bool = True) -> str:
        """Format plan as JSON"""
        return JSONFormatter.format(plan, pretty)

    @staticmethod
    def to_pdf(plan: Dict, output_path: str) -> bool:
        """Format plan as PDF"""
        return PDFFormatter.format(plan, output_path)

    @staticmethod
    def to_html(plan: Dict) -> str:
        """Format plan as HTML"""
        return HTMLFormatter.format(plan)
