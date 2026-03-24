"""
Spillover Network Graph
Converts DY FEVD tables and Granger causality grids into directed NetworkX
graphs and renders them as interactive Plotly figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

from src.data.config import EQUITY_REGIONS, COMMODITY_GROUPS

# ── Asset metadata ──────────────────────────────────────────────────────────

_GROUP_COLORS: dict[str, str] = {
    # Equity regions
    "USA":               "#2980b9",
    "Europe":            "#8e44ad",
    "Japan":             "#e67e22",
    "China":             "#c0392b",
    "India":             "#16a085",
    # Commodity groups
    "Energy":            "#d35400",
    "Precious Metals":   "#CFB991",
    "Industrial Metals": "#7f8c8d",
    "Agriculture":       "#2e7d32",
    "Other":             "#555960",
}

def _asset_group(name: str) -> str:
    for region, assets in EQUITY_REGIONS.items():
        if name in assets:
            return region
    for group, assets in COMMODITY_GROUPS.items():
        if name in assets:
            return group
    return "Other"

def _asset_type(name: str) -> str:
    for assets in EQUITY_REGIONS.values():
        if name in assets:
            return "equity"
    return "commodity"


# ── Graph builders ──────────────────────────────────────────────────────────

def build_dy_graph(table: pd.DataFrame, threshold: float = 3.0) -> nx.DiGraph:
    """
    Build directed graph from DY FEVD spillover table.
    table[i, j] = % of asset i's forecast variance explained by shocks from j.
    Edge j → i added when table[i, j] > threshold.
    Node attrs: group, asset_type, transmit, receive, net.
    """
    G = nx.DiGraph()
    tbl = table.copy()
    np.fill_diagonal(tbl.values, 0)

    transmit = tbl.sum(axis=0)   # col j = total % j transmits to others
    receive  = tbl.sum(axis=1)   # row i = total % i receives from others
    net      = transmit - receive

    for node in tbl.index:
        G.add_node(
            node,
            group=_asset_group(node),
            asset_type=_asset_type(node),
            transmit=float(transmit.get(node, 0)),
            receive=float(receive.get(node, 0)),
            net=float(net.get(node, 0)),
        )

    for i in tbl.index:
        for j in tbl.columns:
            if i != j and tbl.loc[i, j] > threshold:
                G.add_edge(j, i, weight=float(tbl.loc[i, j]))  # j → i

    return G


def build_granger_graph(granger_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph from significant Granger causality results.
    Edge cause → effect, weight = 1 − min_p.
    """
    G = nx.DiGraph()
    sig = granger_df[granger_df["significant"]].copy()
    if sig.empty:
        return G

    for node in set(sig["cause"]) | set(sig["effect"]):
        G.add_node(node, group=_asset_group(node), asset_type=_asset_type(node))

    for _, row in sig.iterrows():
        G.add_edge(
            row["cause"], row["effect"],
            weight=float(1 - row["min_p"]),
            p_value=float(row["min_p"]),
            direction=str(row["direction"]),
            best_lag=int(row.get("best_lag", 1)),
        )

    return G


# ── Layout ──────────────────────────────────────────────────────────────────

def _bipartite_pos(
    equity_nodes: list[str],
    commodity_nodes: list[str],
    radius: float = 1.8,
) -> dict[str, tuple[float, float]]:
    """Equities on left arc, commodities on right arc."""
    pos: dict[str, tuple[float, float]] = {}

    n_eq = len(equity_nodes)
    for i, node in enumerate(equity_nodes):
        frac  = (i + 0.5) / max(n_eq, 1)
        angle = np.pi * (0.1 + 0.8 * frac)
        pos[node] = (np.cos(angle) * radius, np.sin(angle) * radius)

    n_cmd = len(commodity_nodes)
    for i, node in enumerate(commodity_nodes):
        frac  = (i + 0.5) / max(n_cmd, 1)
        angle = np.pi * (-0.1 - 0.8 * frac)
        pos[node] = (np.cos(angle) * radius, np.sin(angle) * radius)

    return pos


def _get_positions(
    G: nx.DiGraph, layout: str = "bipartite"
) -> dict[str, tuple[float, float]]:
    if layout == "bipartite":
        eq_nodes  = [n for n in G.nodes if G.nodes[n].get("asset_type") == "equity"]
        cmd_nodes = [n for n in G.nodes if G.nodes[n].get("asset_type") == "commodity"]
        return _bipartite_pos(eq_nodes, cmd_nodes)
    if layout == "spring":
        return nx.spring_layout(G, seed=42, k=2.2, iterations=120)
    if layout == "circular":
        return nx.circular_layout(G)
    return nx.spring_layout(G, seed=42)


# ── Shared drawing helpers ──────────────────────────────────────────────────

def _edge_annotations(
    edges: list[tuple],
    pos: dict,
    max_w: float,
    color_fn,          # callable(src, data) → hex color
    opacity_scale: tuple[float, float] = (0.2, 0.7),
) -> tuple[list[go.Scatter], list[dict]]:
    """Build edge line traces + arrowhead annotations."""
    traces: list[go.Scatter] = []
    annots: list[dict] = []
    lo, hi = opacity_scale

    for src, tgt, data in edges:
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w      = data.get("weight", 1)
        alpha  = lo + (hi - lo) * min(w / max(max_w, 1e-6), 1.0)
        width  = 0.6 + 3.0 * min(w / max(max_w, 1e-6), 1.0)
        col    = color_fn(src, data)

        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=col),
            opacity=alpha,
            hoverinfo="skip",
            showlegend=False,
        ))
        ax = x0 + 0.82 * (x1 - x0)
        ay = y0 + 0.82 * (y1 - y0)
        annots.append(dict(
            x=x1, y=y1, ax=ax, ay=ay,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=0.9,
            arrowwidth=max(1.0, width * 0.55),
            arrowcolor=col,
            opacity=min(alpha + 0.2, 1.0),
        ))

    return traces, annots


def _node_traces(
    G: nx.DiGraph,
    pos: dict,
    size_fn,      # callable(node, data) → float
    hover_fn,     # callable(node, data) → str
) -> list[go.Scatter]:
    groups: dict[str, dict] = {}
    for node, data in G.nodes(data=True):
        grp = data.get("group", "Other")
        if grp not in groups:
            groups[grp] = {"x": [], "y": [], "size": [], "label": [], "hover": []}
        x, y = pos.get(node, (0, 0))
        groups[grp]["x"].append(x)
        groups[grp]["y"].append(y)
        groups[grp]["size"].append(size_fn(node, data))
        groups[grp]["label"].append(node.replace(" ", "<br>"))
        groups[grp]["hover"].append(hover_fn(node, data))

    traces = []
    for grp, d in groups.items():
        col = _GROUP_COLORS.get(grp, "#555960")
        traces.append(go.Scatter(
            x=d["x"], y=d["y"],
            mode="markers+text",
            marker=dict(size=d["size"], color=col,
                        line=dict(color="#ffffff", width=1.5), opacity=0.93),
            text=d["label"],
            textposition="top center",
            textfont=dict(size=8, family="JetBrains Mono, monospace", color="#000"),
            hovertext=d["hover"],
            hoverinfo="text",
            name=grp,
        ))
    return traces


def _base_layout(height: int, title: str, annotations: list) -> dict:
    return dict(
        template="purdue",
        title=dict(text=title, font=dict(size=13)),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v", x=1.01, y=1, font=dict(size=9),
            bgcolor="rgba(20,22,32,0.85)",
            bordercolor="rgba(150,150,160,0.3)", borderwidth=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showspikes=False, range=[-2.6, 2.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showspikes=False, range=[-2.6, 2.6]),
        margin=dict(l=20, r=180, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
    )


# ── DY network ──────────────────────────────────────────────────────────────

def plot_dy_network(
    G: nx.DiGraph,
    title: str = "Diebold-Yilmaz Spillover Network",
    layout: str = "bipartite",
    top_edges: int = 28,
    height: int = 640,
) -> go.Figure:
    """
    Node size ∝ |net spillover score|.
    Edge thickness ∝ spillover %. Directed transmitter → receiver.
    """
    if not G.nodes:
        return go.Figure()

    pos = _get_positions(G, layout)
    edges_sorted = sorted(
        G.edges(data=True), key=lambda e: e[2].get("weight", 0), reverse=True
    )[:top_edges]
    max_w = max((e[2].get("weight", 1) for e in edges_sorted), default=1)

    def _col(src, data):
        return _GROUP_COLORS.get(G.nodes[src].get("group", "Other"), "#CFB991")

    edge_traces, annots = _edge_annotations(edges_sorted, pos, max_w, _col)

    def _size(node, data):
        return 13 + min(abs(data.get("net", 0)) * 1.1, 26)

    def _hover(node, data):
        return (
            f"<b>{node}</b><br>"
            f"Group: {data.get('group','')}<br>"
            f"Transmits: {data.get('transmit', 0):.1f}%<br>"
            f"Receives:  {data.get('receive', 0):.1f}%<br>"
            f"Net:       {data.get('net', 0):+.1f}%"
        )

    node_traces = _node_traces(G, pos, _size, _hover)
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(**_base_layout(height, title, annots))
    return fig


# ── Net transmitter bar chart ───────────────────────────────────────────────

def plot_net_transmitter_bar(G: nx.DiGraph, height: int = 320) -> go.Figure:
    """Horizontal bar chart: net spillover score per asset (transmit − receive)."""
    rows = [
        {"asset": n, "net": d.get("net", 0), "group": d.get("group", "Other")}
        for n, d in G.nodes(data=True)
    ]
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows).sort_values("net")

    colors = [
        (_GROUP_COLORS.get(r["group"], "#555960") if r["net"] >= 0 else "#c0392b")
        for _, r in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        y=df["asset"],
        x=df["net"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=df["net"].map(lambda v: f"{v:+.1f}%"),
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono, monospace"),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#ABABAB", width=1, dash="dot"))
    fig.update_layout(
        template="purdue",
        height=height,
        title=dict(text="Net Spillover Score (Transmit − Receive)", font=dict(size=11)),
        xaxis=dict(title="Net %", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=9, family="JetBrains Mono, monospace")),
        margin=dict(l=130, r=60, t=45, b=30),
    )
    return fig


# ── Granger network ─────────────────────────────────────────────────────────

_DIR_COLORS = {
    "Commodity → Equity": "#c0392b",
    "Equity → Commodity": "#2980b9",
}

def plot_granger_network(
    G: nx.DiGraph,
    title: str = "Granger Causality Network",
    layout: str = "bipartite",
    height: int = 640,
) -> go.Figure:
    """
    Red edges = commodity → equity, blue = equity → commodity.
    Node size ∝ out-degree (assets it Granger-causes).
    """
    if not G.nodes:
        return go.Figure()

    pos  = _get_positions(G, layout)
    edges = list(G.edges(data=True))
    max_w = max((e[2].get("weight", 1) for e in edges), default=1)

    def _col(src, data):
        return _DIR_COLORS.get(data.get("direction", ""), "#CFB991")

    edge_traces, annots = _edge_annotations(edges, pos, max_w, _col, (0.15, 0.65))

    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values(), default=1)

    def _size(node, data):
        return 12 + min(out_deg.get(node, 0) * 3.5, 26)

    def _hover(node, data):
        return (
            f"<b>{node}</b><br>"
            f"Group: {data.get('group','')}<br>"
            f"Causes: {G.out_degree(node)} assets<br>"
            f"Caused by: {G.in_degree(node)} assets"
        )

    node_traces = _node_traces(G, pos, _size, _hover)

    # Legend entries for edge directions
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color=col, width=2), name=label, showlegend=True)
        for label, col in _DIR_COLORS.items()
    ]

    fig = go.Figure(data=edge_traces + node_traces + legend_traces)
    fig.update_layout(**_base_layout(height, title, annots))
    return fig
