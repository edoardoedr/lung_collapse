"""Shared pieces of the three figures: palette, header, small drawing helpers."""
import math

GREY = dict(fill="#f1efe8", stroke="#5f5e5a", title="#444441", text="#5f5e5a")
GREEN = dict(fill="#e1f5ee", stroke="#0f6e56", title="#085041", text="#0f6e56")
ORANGE = dict(fill="#faece7", stroke="#993c1d", title="#712b13", text="#993c1d")
BLUE = dict(fill="#e6f1fb", stroke="#185fa5", title="#0c447c", text="#185fa5")
PURPLE = dict(fill="#eeedfe", stroke="#534ab7", title="#3c3489", text="#534ab7")
AMBER = dict(fill="#faeeda", stroke="#854f0b", title="#633806", text="#854f0b")


def header(w, h, title, desc):
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="100%%" viewBox="0 0 %d %d" role="img">
<title>%s</title>
<desc>%s</desc>
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
  .t  { font-size: 14px; font-weight: 600; text-anchor: middle; dominant-baseline: central; }
  .s  { font-size: 11.5px; text-anchor: middle; dominant-baseline: central; }
  .c  { font-size: 11px; font-family: Menlo, Consolas, monospace; text-anchor: middle; dominant-baseline: central; }
  .h  { font-size: 13px; font-weight: 600; dominant-baseline: central; }
  .l  { font-size: 11.5px; dominant-baseline: central; fill: #52514e; }
  .m  { font-size: 13px; font-style: italic; font-family: 'Times New Roman', Times, serif; font-variant-numeric: lining-nums; dominant-baseline: central; }
  .k  { font-size: 10.5px; font-family: Menlo, Consolas, monospace; dominant-baseline: central; fill: #52514e; }
  .a  { fill: none; stroke: #898781; stroke-width: 1.5; }
  .grp { fill: none; stroke: #b4b2ab; stroke-width: 0.8; stroke-dasharray: 5 4; }
</style>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#898781" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
<marker id="ap" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#185fa5" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>
<marker id="ag" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#993c1d" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>
<marker id="ak" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#444441" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>
</defs>
<rect width="%d" height="%d" fill="#ffffff"/>
''' % (w, h, title, desc, w, h)


def box(x, y, w, h, pal, title, lines=(), code=None, dash=False):
    """Rounded box with a title, text lines and an optional code line, vertically centred."""
    out = ['<rect x="%g" y="%g" width="%g" height="%g" rx="8" fill="%s" stroke="%s" stroke-width="0.8"%s/>'
           % (x, y, w, h, pal["fill"], pal["stroke"], ' stroke-dasharray="4 3"' if dash else "")]
    rows = [("t", title, pal["title"])] + [("s", l, pal["text"]) for l in lines] + ([("c", code, pal["text"])] if code else [])
    step = 17
    y0 = y + h / 2 - step * (len(rows) - 1) / 2
    for i, (cls, txt, col) in enumerate(rows):
        out.append('<text x="%g" y="%g" class="%s" fill="%s">%s</text>' % (x + w / 2, y0 + i * step, cls, col, txt))
    return "\n".join(out)


def arrow(x1, y1, x2, y2, marker="arrow", cls="a", extra=""):
    return '<line x1="%g" y1="%g" x2="%g" y2="%g" class="%s" marker-end="url(#%s)"%s/>' % (x1, y1, x2, y2, cls, marker, extra)


def path(d, cls="a", marker="arrow"):
    return '<path d="%s" class="%s" marker-end="url(#%s)"/>' % (d, cls, marker)


def text(x, y, s, cls="l", fill=None, anchor=None, extra=""):
    return '<text x="%g" y="%g" class="%s"%s%s%s>%s</text>' % (
        x, y, cls, ' fill="%s"' % fill if fill else "", ' text-anchor="%s"' % anchor if anchor else "", extra, s)


def sub(expr):
    """'p_0', 'e_A', 'p_{atm}' in a string -> SVG subscripts (tspans)."""
    import re
    def rep(m):
        return '<tspan dy="4" font-size="72%%">%s</tspan><tspan dy="-4" font-size="1">​</tspan>' % (m.group(1) or m.group(2))
    return re.sub(r"_(?:\{([^}]*)\}|([A-Za-z0-9]))", rep, expr)
