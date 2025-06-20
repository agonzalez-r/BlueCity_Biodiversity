import geopandas as gpd
from shapely.geometry import Point

def sample_points_along_line(line, step):
    """
    Return points along a line at uniform 'step' spacing, including start and end.
    """
    n_steps = max(1, int(line.length // step))
    distances = [i * step for i in range(n_steps + 1)]
    if line.length - distances[-1] > step * 0.5:  # if last point not close to end, add end
        distances.append(line.length)
    points = [line.interpolate(distance) for distance in distances]
    return points

def make_buffers(roads, name, step=500, radius=250, tol=250,  target_crs=4326, save=False):
    """
    Generate a grid of circular buffers along roads.

    Args:
        roads: GeoDataFrame of LineStrings (must have a projected CRS in meters)
        name: name of the road network (for saving output)
        step: distance between points along roads (meters)
        radius: radius of each buffer (meters)
        tol: snapping tolerance for deduplication (meters)
        target_crs: CRS string for output (default WGS84/EPSG:4326)
        save: whether to save the output as a GeoJSON file

    Returns:
        GeoDataFrame of buffer polygons in target_crs
    """
    roads = roads.to_crs(epsg=2056)
    points = []
    for geom in roads.geometry:
        if geom.length > 0:
            # n = int(geom.length // step)
            # points.extend([geom.interpolate(i * step) for i in range(n + 1)])
            pts = sample_points_along_line(geom, step)
            points.extend(pts)

    # Snap to grid & dedupe
    snapped = {
        (round(pt.x / tol) * tol, round(pt.y / tol) * tol)
        for pt in points
    }
    unique_pts = [Point(x, y) for x, y in snapped]

    buffers = [pt.buffer(radius) for pt in unique_pts]

    buffers_gdf = gpd.GeoDataFrame(
        {"geometry": buffers}, crs=roads.crs
    ).to_crs(epsg=target_crs)

    if save:
        buffers_gdf.to_file(f"../data/buffers/{name}_buffers_{radius}_{step}_{tol}.geojson", driver="GeoJSON")
        print(f"Saved buffers to road_buffers_{radius}_{step}_{tol}.geojson")
    return buffers_gdf
