from sqlalchemy import create_engine
import pandas as pd
import geopandas as gpd
import ast

engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")

def _parse_lanes(x):
    """
    Parses the 'lanes' attribute from a string or integer to a list of integers.

    Parameters:
        x (str or int): The value representing lanes, e.g., '[4,2,3]', '2', or 2.

    Returns:
        list[int]: A list of integers representing the number of lanes.
                   Returns an empty list if parsing fails.
    """
    try:
        if isinstance(x, str) and x.startswith('['):
            # literal_eval gives you something like ['4','2','3']
            strs = ast.literal_eval(x)
            # convert each element to int
            return [int(v) for v in strs]
        else:
            # a single‐value string like '2'
            return [int(x)]
    except:
        return []


def _parse_speeds(x):
    """
    Parses the 'maxspeed' attribute from a string or integer to a list of integers.

    Parameters:
        x (str or int): The value representing speed limits, e.g., "['80','100']", '80', or 80.

    Returns:
        list[int]: A list of integers representing speed limits.
                   Returns an empty list if parsing fails.
    """
    try:
        if isinstance(x, str) and x.startswith('['):
            # e.g. "['80','100']"
            return [int(v) for v in ast.literal_eval(x)]
        else:
            return [int(x)]
    except:
        return []



def assign_habitats_to_buffers(habitats, buffers):
    """
    Assigns habitats to buffers based on spatial intersections and calculates statistics.

    Parameters:
    habitats (geopandas.GeoDataFrame): GeoDataFrame containing habitat objects with attributes 'id' and 'geometry'.
    buffers (geopandas.GeoDataFrame): GeoDataFrame containing buffer objects with attributes 'habitat_id' and 'geometry'.

    Returns:
    pandas.DataFrame: A DataFrame containing aggregated statistics for each buffer, including:
        - buffer_id: Unique identifier for each buffer.
        - TypoCH: Habitat type identifier.
        - habitat_area: Total area of habitat within the buffer.
        - buffer_area: Total area of the buffer.
        - percentage: Percentage of the buffer area covered by the habitat.
    """
    habitats.to_postgis("habitats", engine, if_exists='replace', index=False)
    buffers.to_postgis("buffers", engine, if_exists='replace', index=False)

    sql = """
    WITH numbered_buffers AS (
      SELECT
        row_number() OVER ()         AS buffer_id,
        geometry
      FROM buffers
    )
    SELECT
      nb.buffer_id,
      h."TypoCH",
      ROUND(SUM(ST_Area(ST_Intersection(h.geometry, nb.geometry)))::numeric, 2) AS habitat_area,
      ROUND(ST_Area(nb.geometry)::numeric,                          2) AS buffer_area,
      ROUND(
        (SUM(ST_Area(ST_Intersection(h.geometry, nb.geometry)))
         / NULLIF(ST_Area(nb.geometry), 0)
        )::numeric
        , 4
      ) AS percentage
    FROM habitats AS h
    JOIN numbered_buffers AS nb
      ON ST_Intersects(h.geometry, nb.geometry)
    GROUP BY nb.buffer_id, h."TypoCH", nb.geometry;
    """

    df_habitats = pd.read_sql_query(sql, engine)

    return df_habitats


def assign_road_data_to_buffers(roads, buffers):
    """
    Assigns road data to buffers based on spatial intersections.

    Parameters:
    roads (geopandas.GeoDataFrame): GeoDataFrame containing road data with attributes such as 'lanes', 'maxspeed', and 'tunnel'.
    buffers (geopandas.GeoDataFrame): GeoDataFrame containing buffer data with attributes 'buffer_id' and 'geometry'.

    Returns:
    geopandas.GeoDataFrame: Aggregated road data per buffer, including:
        - avg_speed: Mean of average speed limits within each buffer.
        - max_lanes: Maximum number of lanes within each buffer.
        - is_tunnel: Boolean indicating if any road within the buffer is a tunnel.
    """
    # Perform spatial join to find roads intersecting buffers
    roads_in_buf = gpd.sjoin(roads, buffers[['buffer_id', 'geometry']], how="inner", predicate="intersects")

    # Parse and calculate lane-related attributes
    roads_in_buf['lanes_list'] = roads_in_buf['lanes'].apply(_parse_lanes)
    roads_in_buf['max_lanes'] = roads_in_buf['lanes_list'].apply(lambda L: max(L) if L else None)

    # Parse and calculate speed-related attributes
    roads_in_buf['speeds_list'] = roads_in_buf['maxspeed'].apply(_parse_speeds)
    roads_in_buf['max_speed_lim'] = roads_in_buf['speeds_list'].apply(lambda L: max(L) if L else None)
    roads_in_buf['avg_speed_lim'] = roads_in_buf['speeds_list'].apply(lambda L: sum(L) / len(L) if L else None)

    # Aggregate road data by buffer_id
    df_roads = roads_in_buf.groupby('buffer_id').agg(
        avg_speed=('avg_speed_lim', 'mean'),
        max_lanes=('max_lanes', 'max'),
        is_tunnel=('tunnel', 'any')
    ).reset_index()

    return df_roads

def assign_rail_data_to_buffers(railway, buffers):

    # Perform spatial join to find railways intersecting buffers
    rail_in_buf = gpd.sjoin(railway, buffers[['buffer_id', 'geometry']], how="inner", predicate="intersects")

    rail_in_buf['speeds_list'] = rail_in_buf['maxspeed'].apply(_parse_speeds)
    rail_in_buf['max_speed_lim'] = rail_in_buf['speeds_list'].apply(lambda L: max(L) if L else None)
    rail_in_buf['avg_speed_lim'] = rail_in_buf['speeds_list'].apply(lambda L: sum(L) / len(L) if L else None)

    # Aggregate rail data by buffer_id
    df_rail = rail_in_buf.groupby('buffer_id').agg(
        avg_speed=('avg_speed_lim', 'mean'),
        max_speed=('max_speed_lim', 'max'),
        is_tunnel=('tunnel', 'any')
    ).reset_index()

    return df_rail