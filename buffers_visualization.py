import pydeck as pdk
import json

def visualize_buffers(
    buffers_gdf,
    html_filename="buffers_visualization.html",
    simplify_tolerance=0.0005,
    view_lat=46.6,
    view_lon=6.5,
    fill_color=None,
    line_color=None,
    open_browser=True,
    notebook_display=False
):
    """
    Visualize a grid GeoDataFrame using Pydeck's GeoJsonLayer and export as HTML.

    Args:
        buffers_gdf: GeoDataFrame (must be in EPSG:4326, WGS84).
        html_filename: Name of the output HTML file.
        simplify_tolerance: Tolerance for simplifying geometries (default 0.0005).
        view_lat, view_lon: Map view starting coordinates.
        fill_color: RGBA fill color for polygons.
        line_color: RGB line color for polygon edges.
        open_browser: If True, open the HTML file in browser after saving.
        notebook_display: If True, display in Jupyter notebook.
    """
    if line_color is None:
        line_color = [0, 0, 0]
    if fill_color is None:
        fill_color = [200, 100, 150, 80]

    # Ensure in EPSG:4326 for web mapping
    # buffers_gdf = buffers_gdf.to_crs(epsg=4326)

    # Simplify geometries for better performance
    # buffers_gdf["geometry"] = buffers_gdf.geometry.simplify(simplify_tolerance)

    data = json.loads(buffers_gdf.to_json())

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=data,
        stroked=True,
        filled=True,
        get_fill_color=fill_color,
        get_line_color=line_color,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=view_lat,
        longitude=view_lon,
        zoom=9,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
    )

    deck.to_html(
        html_filename,
        notebook_display=notebook_display,
        offline=True,
        open_browser=open_browser
    )

