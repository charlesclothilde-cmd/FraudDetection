import math
from pathlib import Path


def write_suspicious_ring_svg(scores, features, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_component = (
        scores.groupby("component_id")["risk_score"]
        .agg(["mean", "count"])
        .assign(weighted=lambda df: df["mean"] * df["count"].clip(upper=20))
        .sort_values("weighted", ascending=False)
        .index[0]
    )
    component = features[features["component_id"] == top_component].copy().head(40)
    width, height = 900, 620
    cx, cy = width / 2, height / 2
    radius = 220

    nodes = []
    for ix, row in enumerate(component.itertuples(index=False)):
        angle = 2 * math.pi * ix / max(len(component), 1)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        risk = min(max(float(getattr(row, "graph_risk_score", 0.0)) / 3.0, 0.0), 1.0)
        color = _risk_color(risk, int(row.is_fraud_ring))
        nodes.append((row.user_id, x, y, color, row.is_fraud_ring, row.component_size))

    lines = []
    for i, left in enumerate(nodes):
        for right in nodes[i + 1 :]:
            if (i + len(right[0])) % 7 == 0 or (i + len(left[0])) % 11 == 0:
                lines.append(
                    '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9aa4b2" stroke-width="1" opacity="0.25" />'
                    % (left[1], left[2], right[1], right[2])
                )

    circles = []
    labels = []
    for user_id, x, y, color, is_fraud, component_size in nodes:
        stroke = "#111827" if is_fraud else "#475569"
        circles.append(
            '<circle cx="%.1f" cy="%.1f" r="14" fill="%s" stroke="%s" stroke-width="2" />'
            % (x, y, color, stroke)
        )
        labels.append(
            '<text x="%.1f" y="%.1f" font-size="10" text-anchor="middle" fill="#111827">%s</text>'
            % (x, y + 31, user_id[-4:])
        )

    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">
<rect width="100%%" height="100%%" fill="#f8fafc"/>
<text x="40" y="42" font-size="24" font-family="Arial" font-weight="700" fill="#0f172a">Top Suspicious Shared-Infrastructure Component</text>
<text x="40" y="72" font-size="14" font-family="Arial" fill="#475569">Dark outlines are injected fraud-ring users. Warmer fills indicate higher graph risk.</text>
<g font-family="Arial">
%s
%s
%s
</g>
</svg>
""" % (
        width,
        height,
        width,
        height,
        "\n".join(lines),
        "\n".join(circles),
        "\n".join(labels),
    )
    output_path.write_text(svg)


def _risk_color(risk, is_fraud):
    if is_fraud:
        return "#ef4444"
    if risk > 0.66:
        return "#f97316"
    if risk > 0.33:
        return "#facc15"
    return "#38bdf8"
