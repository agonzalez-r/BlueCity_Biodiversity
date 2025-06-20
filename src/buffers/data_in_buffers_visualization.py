import pydeck as pdk
import json
from matplotlib import pyplot as plt

# Center in Lausanne, Switzerland
view_lat = 46.550
view_lon = 6.633
zoom = 12

def _create_pydeck_visualization(geojson_data, fields_to_show, html_filename, color_property="properties.color"):
    tooltip_html = "<br>".join([f"<b>{col}:</b> {{{col}}}" for col in fields_to_show])

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_data,
        stroked=True,
        filled=True,
        get_fill_color=color_property,
        get_line_color=[0, 0, 0, 100],
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        type="MapView",
        latitude=view_lat,
        longitude=view_lon,
        zoom=zoom,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        map_style='light',
        initial_view_state=view_state,
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "steelblue", "color": "white"}},
    )

    deck.to_html(html_filename, notebook_display=False, open_browser=True)

def visualize_buffers_habitat_data(buffers, html_filename="habitat_buffers_visualization.html"):
    """
    Visualize buffer polygons colored by dominant habitat type using pydeck.

    Parameters
    ----------
    buffers : GeoDataFrame
        Buffers data, with columns: 'buffer_id', 'geometry', 'avg_speed', 'max_lanes', 'is_tunnel',
        'fragmentation_index', 'dominant_typoch', 'dominant_habitat_name', 'color'
    html_filename : str, optional
        Output HTML file name for visualization (default: "habitat_buffers_visualization.html").

    The function displays all attributes on hover except index/geometry/color.
    """
    buffers = buffers.to_crs(epsg=4326)
    geojson_data = json.loads(buffers.to_json())
    fields_to_show = [col for col in buffers.columns if col not in ['geometry', 'index', 'color']]

    _create_pydeck_visualization(geojson_data, fields_to_show, html_filename)

def visualize_buffers_data(buffers, feature, html_filename="buffers_visualization.html"):
    """
    Visualize buffer polygons colored by a specific feature using pydeck.

    Parameters
    ----------
    buffers : GeoDataFrame
        Buffers data, with columns: 'buffer_id', 'geometry', 'avg_speed', 'max_lanes', 'is_tunnel',
        'fragmentation_index', 'dominant_typoch', 'dominant_habitat_name', 'color'
    feature : str
        The column name in buffers to use for coloring the polygons.
    html_filename : str, optional
        Output HTML file name for visualization (default: "buffers_visualization.html").

    The function displays all attributes on hover except index/geometry/color.
    """
    buffers_ = buffers.to_crs(epsg=4326).copy()
    vals = buffers_[feature].astype(float).fillna(0)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin == 0:
        normalized = vals.apply(lambda _: 0.5)
    else:
        normalized = (vals - vmin) / (vmax - vmin)

    cmap = plt.get_cmap('Reds')

    def norm_to_rgba(val_norm):
        r, g, b, _ = cmap(val_norm)
        return [int(r * 255), int(g * 255), int(b * 255), 100]

    buffers_['color'] = normalized.apply(norm_to_rgba)
    geojson_data = json.loads(buffers_.to_json())
    fields_to_show = [col for col in buffers_.columns if col not in ['geometry', 'index', 'color']]

    _create_pydeck_visualization(geojson_data, fields_to_show, html_filename)