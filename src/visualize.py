import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_contour_map(lat, lon, values, cmap='viridis', vmin=None, vmax=None, clev=11, title='', save_path=None):
    """
    Plot a contour map with latitude, longitude, and values on a global map.

    Parameters:
    - lat: Array-like, latitude values.
    - lon: Array-like, longitude values.
    - values: Array-like, data values corresponding to lat/lon.
    - cmap: Colormap for the plot. Default is 'viridis'.
    - vmin: Minimum value for the colormap. Default is min(values).
    - vmax: Maximum value for the colormap. Default is max(values).
    - clev: Number of contour levels. Default is 11.
    - title: Title of the plot.
    - save_path: Path to save the plot. If None, the plot is shown interactively.
    """
    # Set up the plot
    lon = np.mod(lon + 180, 360) - 180  # Convert 0–360 to -180–180

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(10, 5)
    )
    
    # Set global map features
    ax.set_global()
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Handle colormap limits and contour levels
    vmin = vmin if vmin is not None else np.min(values)
    vmax = vmax if vmax is not None else np.max(values)
    clevels = np.linspace(vmin, vmax, clev)

    # Plot the contour map
    contour = ax.tricontourf(
        lon, lat, values, levels=clevels, cmap=cmap, transform=ccrs.PlateCarree()
    ) #That showed a white space for some reason, lets try with contourf:
    # contour = ax.contourf(
    #     lon, lat, values, levels=clevels, cmap=cmap, transform=ccrs.PlateCarree()
    # )

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label('Value')

    # Add title
    if title != '':
        ax.set_title(title, fontsize=14)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    # close completely:
    plt.clf()