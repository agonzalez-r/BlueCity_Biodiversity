import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def get_habitat_name(names, typoch):
    """
    Get the English name of the habitat for a given TypoCH code.

    Parameters
    ----------
    names : DataFrame
        DataFrame with columns 'normalized_typo' and 'anglais'.
    typoch : str or int
        TypoCH code to look up.

    Returns
    -------
    str or None
        English name of the habitat, or None if not found.
    """
    typoch_lookup = dict(zip(names['normalized_typo'], names['anglais']))
    return typoch_lookup.get(str(typoch), None)

def most_common_habitat(habitats, typoch_names):
    """
    Print the 10 most common habitats by total area.

    Parameters
    ----------
    habitats : DataFrame
        DataFrame with columns 'TypoCH' and 'habitat_area'.
    typoch_names : DataFrame
        DataFrame with habitat code and English name mapping.
    """
    habitat_totals = habitats.groupby('TypoCH')['habitat_area'].sum()
    most_common = habitat_totals.sort_values(ascending=False).head(10)
    print("\n----- 10 Most common habitats by total area -----\n")
    for typoch in most_common.items():
        name = get_habitat_name(names=typoch_names, typoch=typoch[0])
        print(f"{typoch[0]}: {name}, {typoch[1]:,.0f} m²")

def get_dominant_habitats(habitats):
    """
    Get the dominant habitat (by area) for each buffer.

    Parameters
    ----------
    habitats : DataFrame
        DataFrame with columns 'buffer_id', 'TypoCH', and 'habitat_area'.

    Returns
    -------
    DataFrame
        DataFrame with columns 'buffer_id' and 'dominant_typoch'.
    """
    dominant_habitat = habitats.loc[habitats.groupby('buffer_id')['habitat_area'].idxmax()]
    dominant_habitat = dominant_habitat[['buffer_id', 'TypoCH']].rename(columns={'TypoCH': 'dominant_typoch'})
    return dominant_habitat

def create_fragmentation_index(habitats):
    """
    Calculate the fragmentation index (number of unique TypoCH) per buffer.

    Parameters
    ----------
    habitats : DataFrame
        DataFrame with columns 'buffer_id' and 'TypoCH'.

    Returns
    -------
    DataFrame
        DataFrame with columns 'buffer_id' and 'fragmentation_index'.
    """
    fragmentation_index = habitats.groupby('buffer_id')['TypoCH'].nunique().reset_index()
    fragmentation_index = fragmentation_index.rename(columns={'TypoCH': 'fragmentation_index'})
    return fragmentation_index

def create_habitat_palette(dom):
    """
    Generate a color palette for TypoCH habitat codes.

    Parameters
    ----------
    dom : iterable
        Iterable of TypoCH codes.

    Returns
    -------
    dict
        Mapping from TypoCH code to hex color.
    """
    base_colors = {
        '6': '#228B22',  # forest green
    }
    gray_hex = '#808080'
    base6_hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base_colors['6']))
    group_hues = {
        '8': 0.9,  # pink/magenta
        '4': 0.08,  # orange
        '5': 0.15,  # yellow
        '1': mcolors.rgb_to_hsv(mcolors.to_rgb('#1E90FF'))[0],  # dodger blue
    }
    default_hue = mcolors.rgb_to_hsv(mcolors.to_rgb(gray_hex))[0]

    def get_group(typoch):
        s = str(typoch)
        if s.startswith("6000"):
            return '6'
        elif '.' in s:
            return s.split('.')[0]
        else:
            try:
                return str(int(float(s)))
            except Exception:
                return s

    group_to_subcats = defaultdict(list)
    for typoch in dom:
        group = get_group(typoch)
        group_to_subcats[group].append(typoch)

    habitat_palette = {}
    for group, subcats in group_to_subcats.items():
        n = len(subcats)
        if group == '8':
            hue = group_hues['8']
            sat_min, sat_max = 0.6, 1.0
            val_min, val_max = 0.8, 1.0
            for idx, typoch in enumerate(sorted(subcats)):
                sat = sat_min + (sat_max - sat_min) * idx / max(n - 1, 1)
                val = val_min + (val_max - val_min) * (n - 1 - idx) / max(n - 1, 1)
                rgb = mcolors.hsv_to_rgb([hue, sat, val])
                habitat_palette[typoch] = mcolors.to_hex(rgb)
        elif group == '6':
            fixed_colors = {
                '6000.0': '#006400',  # dark green
                '6000.1': '#32CD32',  # lime green
            }
            for typoch in sorted(subcats):
                if str(typoch) in fixed_colors:
                    habitat_palette[typoch] = fixed_colors[str(typoch)]
                else:
                    idx = sorted(subcats).index(typoch)
                    hue_shift = (idx / max(n, 1)) * 0.2
                    value_shift = 0.15 * ((idx % 2) - 0.5)
                    h = (base6_hsv[0] + hue_shift) % 1.0
                    s = min(1.0, base6_hsv[1] + 0.08 * ((idx // 2) % 2))
                    v = min(1.0, max(0.3, base6_hsv[2] + value_shift))
                    rgb = mcolors.hsv_to_rgb([h, s, v])
                    habitat_palette[typoch] = mcolors.to_hex(rgb)
        elif group == '9':
            for typoch in subcats:
                habitat_palette[typoch] = gray_hex
        else:
            hue = group_hues.get(group, default_hue)
            for idx, typoch in enumerate(sorted(subcats)):
                sat = 0.6 + 0.4 * idx / max(n - 1, 1)
                val = 0.8 + 0.2 * ((idx % 2))
                rgb = mcolors.hsv_to_rgb([hue, sat, val])
                habitat_palette[typoch] = mcolors.to_hex(rgb)
    return habitat_palette

def hex_to_rgba(hex_color, alpha=100):
    """
    Convert a hex color string to an [R, G, B, A] list.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., '#RRGGBB').
    alpha : int, optional
        Alpha value (0–255), default is 100.

    Returns
    -------
    list
        List of [R, G, B, A] values.
    """
    rgb = [int(255 * x) for x in mcolors.to_rgb(hex_color)]
    return rgb + [alpha]

def create_pie_chart(data, habitat_palette, typoch_names, name):
    """
    Plot a pie chart showing distribution in `data`, merging categories <2% into 'others'.

    Parameters
    ----------
    data : pandas Series or list-like
        Series of habitat codes (or any categorical data) for which to plot distribution.
    habitat_palette : dict
        Mapping from TypoCH code to hex color.
    typoch_names : DataFrame
        DataFrame with habitat code and English name mapping.
    name : str
        Title for the plot.
    """
    counts = data.value_counts()
    total = counts.sum()
    percentages = counts / total
    main = percentages[percentages >= 0.02]
    other = percentages[percentages < 0.02]
    if not other.empty:
        main['others'] = other.sum()
    labels = [get_habitat_name(typoch_names, typoch) for typoch in main.index.tolist()]
    sizes = main.values
    colors = [habitat_palette.get(l, "#B0B0B0") if l != 'others' else "#B0B0B0" for l in main.index.tolist()]
    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90
    )
    plt.axis('equal')
    plt.title(name)
    plt.show()

def cluster_with_undersampling(buffers, n_clusters=4, max_per_combo=50, random_state=0):
    """
    Perform KMeans clustering on (avg_speed, max_lanes) with undersampling
    of over-represented (avg_speed, max_lanes) pairs.

    Parameters
    ----------
    buffers : DataFrame or GeoDataFrame
        Must contain columns 'avg_speed', 'max_lanes', and 'is_tunnel'.
    n_clusters : int, optional
        Number of KMeans clusters to find among non-tunnel roads.
    max_per_combo : int, optional
        Maximum number of rows to sample per unique (avg_speed, max_lanes) pair.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    DataFrame
        DataFrame with a new column 'road_subcat' assigned as:
        - 'Tunnel' for rows where is_tunnel == True
        - an integer 0..(n_clusters-1) for non-tunnel rows (their cluster label)
    ndarray
        Cluster centers from KMeans.
    """
    is_tun = buffers['is_tunnel'] == True
    non_tunnel = buffers.loc[~is_tun].copy()
    non_tunnel['_combo'] = list(zip(non_tunnel['avg_speed'].fillna(0),
                                    non_tunnel['max_lanes'].fillna(0)))
    sampled_idx = []
    rng = np.random.RandomState(random_state)
    for combo, group in non_tunnel.groupby('_combo'):
        count = len(group)
        if count <= max_per_combo:
            sampled_idx.extend(group.index.tolist())
        else:
            chosen = rng.choice(group.index, size=max_per_combo, replace=False)
            sampled_idx.extend(chosen.tolist())
    sampled = non_tunnel.loc[sampled_idx].copy()
    X_sampled = sampled[['avg_speed', 'max_lanes']].fillna(0).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    sampled['road_subcat'] = kmeans.fit_predict(X_sampled)
    full_X = non_tunnel[['avg_speed', 'max_lanes']].fillna(0).values
    non_tunnel['road_subcat'] = kmeans.predict(full_X)
    non_tunnel.drop(columns=[col for col in buffers.columns if col == '_combo'], inplace=True, errors='ignore')
    return non_tunnel, kmeans.cluster_centers_

def plot_fragmentation_dominant_type(buffers, habitat_palette):
    """
    Plot fragmentation index vs. dominant TypoCH habitat type.

    Parameters
    ----------
    buffers : DataFrame
        DataFrame with columns 'dominant_typoch' and 'fragmentation_index'.
    habitat_palette : dict
        Mapping from TypoCH code to hex color.

    Returns
    -------
    matplotlib.pyplot
        The plot object.
    """
    unique_types = sorted(buffers['dominant_typoch'].unique())
    type_to_x = {typ: idx for idx, typ in enumerate(unique_types)}
    x_positions = buffers['dominant_typoch'].map(type_to_x)
    colors = buffers['dominant_typoch'].map(lambda t: habitat_palette.get(t, "#B0B0B0"))
    plt.figure(figsize=(10, 8))
    plt.scatter(
        x_positions,
        buffers['fragmentation_index'],
        c=colors,
        alpha=0.7,
        edgecolor='k'
    )
    plt.xlabel('Dominant TypoCH Habitat Type')
    plt.ylabel('Fragmentation Index')
    plt.title('Fragmentation Index vs. Dominant Habitat Type')
    plt.xticks(
        range(len(unique_types)),
        unique_types,
        rotation=90
    )
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_habitat_geometries(buffer_id, gdf, buffers):
    """
    Plot the intersected habitat geometries for a specific buffer.

    Parameters:
        buffer_id (int): ID of the buffer to visualize.
        gdf (GeoDataFrame): GeoDataFrame containing habitat geometries with a 'buffer_id' column and 'color' column.
        buffers (GeoDataFrame): GeoDataFrame containing buffer geometries with a 'buffer_id' column.
    """
    # Filter for the specific buffer
    gdf_one_buffer = gdf[gdf["buffer_id"] == buffer_id]

    # Plot the intersected habitat geometries
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    gdf_one_buffer.plot(ax=ax, color=gdf_one_buffer['color'])

    # Add buffer outline on top
    buffers[buffers['buffer_id'] == buffer_id].boundary.plot(ax=ax, color='black', linewidth=1)

    ax.set_axis_off()
    plt.show()

def plot_habitat_pie(habitats_buffers, buffer_id, palette, typoch_names):
    """
    Plot a pie chart showing habitat composition in a specific buffer.

    Parameters:
        habitats_buffers (DataFrame): DataFrame with buffer_id, TypoCH, percentage
        buffer_id (int): ID of the buffer to visualize
        palette (dict): Mapping from TypoCH code to hex color
        typoch_names (dict): Mapping from TypoCH code to human-readable name
    """
    # Filter for buffer
    data = habitats_buffers[habitats_buffers['buffer_id'] == buffer_id]

    # Get percentage by TypoCH
    percentages = data.set_index('TypoCH')['percentage']

    # Split main vs. others
    main = percentages[percentages >= 0.02].copy()
    other = percentages[percentages < 0.02]

    if not other.empty:
        main.loc['others'] = other.sum()

    # Prepare labels
    labels = [
        f'{typoch}: {get_habitat_name(typoch_names,typoch)}' if typoch != 'others' else 'Others'
        for typoch in main.index
    ]

    # Prepare sizes
    sizes = main.values

    # Prepare colors
    colors = [
        palette.get(typoch, "#B0B0B0") if typoch != 'others' else "#B0B0B0"
        for typoch in main.index
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')  # figure background
    wedges, texts, _ = plt.pie(
        sizes,
        # labels=labels,
        colors=colors,
        startangle=90,
        autopct='%1.1f%%',
    )
    plt.axis('equal')
    plt.show()

    # Show legend
    fig_legend, ax_legend = plt.subplots(figsize=(6, 6), facecolor='black')
    ax_legend.axis('off')  # Hide axes

    # Create handles
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        for label, color in zip(labels, colors)
    ]

    # Add legend
    legend = ax_legend.legend(
        handles=legend_handles,
        loc='center',
        title="Habitat Types",
        title_fontsize='large',
        frameon=True
    )

    # Set legend background and text color
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')

    # Set text color to white
    for text in legend.get_texts():
        text.set_color('white')
    legend.get_title().set_color('white')

    plt.show()
