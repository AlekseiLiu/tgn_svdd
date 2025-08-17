import numpy as np


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import gaussian_kde



def plot_array(array, labels, save_path, title='title', save=False, save_name='_',
               alpha_0=0.9, alpha_1=0.6, size_0=0.5, size_1=1):
    sizes = [size_0 if label == 0 else size_1 for label in labels]
    alpha = [alpha_0 if label == 0 else alpha_1 for label in labels]
    plt.scatter(range(len(array)), array, c=labels, cmap='viridis', alpha=alpha, s=sizes)
    mean0 = np.mean([array[i] for i in range(len(array)) if labels[i] == 0])
    mean1 = np.mean([array[i] for i in range(len(array)) if labels[i] == 1])
    plt.axhline(mean0, color='red', linestyle='--', label='Mean of label 0')
    plt.axhline(mean1, color='blue', linestyle='-.', label='Mean of label 1')
    plt.legend()
    plt.title(title)
    # plt.ylim(0, np.median(array)*30)#max(array)*1.05)
    if save == True:
        plt.savefig(save_path+f'/epoch_{save_name}.png', dpi=300)
    plt.show()
    
    
    
    
def plot_nodes_color(array, labels, save_path, title='title', save=False, save_name='_',
                alpha_0=0.05, alpha_1=0.8, size_0=0.1, size_1=1.5):
    labels = [int(x) for x in labels]
    sizes = [size_0 if label == 0 else size_1 for label in labels]
    alpha = [alpha_0 if label == 0 else alpha_1 for label in labels]
    colors = [(0, 0, 139/255), 'yellow', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    # plt.scatter(range(len(array)), array, c=colors[labels], alpha=alpha, s=sizes)
    plt.scatter(range(len(array)), array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
    means = [np.mean([array[i] for i in range(len(array)) if labels[i] == j]) for j in range(6)]
    classes = ['norm class', 'attack', '(32, *)', '(*, 32)', '(11, *)', '(*, 11)']
    for j in range(0, 6):
        plt.axhline(means[j], color=colors[j], linestyle='--', label=classes[j])
    # plt.axhline(means[0], color=(0, 0, 139/255), linestyle='--', label=classes[0])
    plt.legend(fontsize='small')
    plt.title(title)
    if save:
        plt.savefig(f'{save_path}/epoch_{save_name}.png', dpi=300)
    plt.show()




def plot_nodes_color_kde_precentil_3(arrays, lables_arr, save_path, title='title', kde_title='kde', 
                                     percentile=99, save=False, save_name='_',
                                     alpha_0=0.05, alpha_1=0.8, size_0=0.1, size_1=1.5, array_x=None):
    
    n = len(arrays)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(n * 6, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    
    cmap = plt.get_cmap('viridis')
    # Get the dark blue color (minimum value of the colormap)
    viridis_dark_blue = cmap(0)
    # Get the yellow color (maximum value of the colormap)
    viridis_yellow = cmap(0.98)
    
    
    
    y_max = max([np.max(array) for array in arrays])
    kde_colors = ['r', 'b', 'g']
    kde_labels = ['kde train', 'kde val', 'kde test']
    threshold_99 = None
    
    for idx, (array, labels) in enumerate(zip(arrays, lables_arr)):
        labels = [int(x) for x in labels]
        sizes = [size_0 if label == 0 else size_1 for label in labels]
        alpha = [alpha_0 if label == 0 else alpha_1 for label in labels]
        # colors = [(0, 0, 139/255), 'yellow', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        colors = [viridis_dark_blue, viridis_yellow, 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        
        if array_x is not None:
            axes[0, idx].scatter(array_x[idx], array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
            # Create evenly spaced x-axis ticks
            num_ticks = 10
            min_x, max_x = np.min(array_x[idx]), np.max(array_x[idx])
            xticks = np.linspace(min_x, max_x, num_ticks)
            
            axes[0, idx].set_xticks(xticks)
            # axes[0, idx].set_xticklabels([str(x) for x in array_x[idx]], rotation=90)
        else:
            axes[0, idx].scatter(range(len(array)), array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
            interval = round(len(array) / 10)
            axes[0, idx].set_xticks(range(0, len(array), interval))
            # axes[0, idx].set_xticklabels([str(i) for i in range(len(array))], rotation=90)
        
        means = [np.mean([array[i] for i in range(len(array)) if labels[i] == j]) for j in range(6)]
        classes = ['norm class', 'attack', '(32, *)', '(*, 32)', '(11, *)', '(*, 11)']
        
        for j in range(0, 6):
            axes[0, idx].axhline(means[j], color=colors[j], linestyle='--', label=classes[j])
        # axes[0, idx].legend(fontsize='small')
        axes[0, idx].set_title(title)
        
        if idx == 0:
            threshold_99 = np.percentile(array, percentile)
        axes[0, idx].axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
        
        axes[0, idx].legend(fontsize='small')
        axes[0, idx].set_ylim(0, y_max * 1.05)
        
        # y_values = np.array(array)
        # kde = gaussian_kde(y_values)
        # y_range = np.linspace(0, np.max(y_values), 100)
        # pdf_kde = kde.evaluate(y_range)

        # axes[1, idx].plot(y_range, pdf_kde, 'r-', lw=2)
        # axes[1, idx].
        for k, arr in enumerate(arrays):
            y_values = np.array(arr)
            kde = gaussian_kde(y_values)
            y_range = np.linspace(0, np.max(y_values), 100)
            pdf_kde = kde.evaluate(y_range)
            
            if k == idx:
                axes[1, idx].plot(y_range, pdf_kde, kde_colors[k] + '-', lw=2, label=kde_labels[k])
            else:
                axes[1, idx].plot(y_range, pdf_kde, kde_colors[k] + '-', lw=2, alpha=0.3, label=kde_labels[k])
                
        axes[1, idx].legend(fontsize='small')

        axes[1, idx].set_xlabel(kde_title)

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_path}/epoch_{save_name}.png', dpi=300)
    plt.show()



def plot_nodes_color_kde_precentil_binary(arrays, lables_arr, save_path, title=['Train distances', 'Validation distances', 'Test distances'], kde_title=['KDE train', 'KDE validation', 'KDE test'],
                                     percentile=99, save=False, save_name='_',
                                     alpha_0=0.05, alpha_1=0.8, size_0=0.1, size_1=1.5,
                                     array_x=None, metric_name=None, metric_value=0):
    
    n = len(arrays)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(n * 6, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    
    cmap = plt.get_cmap('viridis')
    # Get the dark blue color (minimum value of the colormap)
    viridis_dark_blue = cmap(0)
    # Get the yellow color (maximum value of the colormap)
    viridis_yellow = cmap(0.98)
    
    
    
    y_max = max([np.max(array) for array in arrays])
    kde_colors = ['r', 'b', 'g']
    kde_labels = ['kde train', 'kde validation', 'kde test']
    threshold_99 = None
    
    caption_labels = ['Normal', 'Attack']
    
    for idx, (array, labels) in enumerate(zip(arrays, lables_arr)):
        labels = [int(x) for x in labels]
        sizes = [size_0 if label == 0 else size_1 for label in labels]
        alpha = [alpha_0 if label == 0 else alpha_1 for label in labels]
        # colors = [(0, 0, 139/255), 'yellow', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        colors = [viridis_dark_blue, viridis_yellow]
        
        if array_x is not None:
            axes[0, idx].scatter(array_x[idx], array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
            # Create evenly spaced x-axis ticks
            num_ticks = 10
            min_x, max_x = np.min(array_x[idx]), np.max(array_x[idx])
            xticks = np.linspace(min_x, max_x, num_ticks)
            
            axes[0, idx].set_xticks(xticks)
            # axes[0, idx].set_xticklabels([str(x) for x in array_x[idx]], rotation=90)
        else:
            axes[0, idx].scatter(range(len(array)), array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
            interval = round(len(array) / 10)
            axes[0, idx].set_xticks(range(0, len(array), interval))
            # axes[0, idx].set_xticklabels([str(i) for i in range(len(array))], rotation=90)
        
        means = [np.mean([array[i] for i in range(len(array)) if labels[i] == j]) for j in range(6)]
        classes = ['Normal mean', 'Attack mean']
        
        for j in range(0, 2):
            axes[0, idx].axhline(means[j], color=colors[j], linestyle='--', label=classes[j])
        # axes[0, idx].legend(fontsize='small')
        axes[0, idx].set_title(title[idx])
        
        if idx == 0:
            threshold_99 = np.percentile(array, percentile)
        axes[0, idx].axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
        
        legend_elements = axes[0, idx].get_legend_handles_labels()[0]
        legend_labels = axes[0, idx].get_legend_handles_labels()[1]
        new_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=caption_labels[i]) for i in range(len(caption_labels))]
        axes[0, idx].legend(handles=legend_elements + new_elements, labels=legend_labels + caption_labels,fontsize='small')
        
               
        
        # axes[0, idx].legend(fontsize='small')
        axes[0, idx].set_ylim(0, y_max * 1.05)
        axes[0, idx].set_xlabel('time, ms')
        axes[0, idx].set_ylabel('distance from ball center')
        
       
        for k, arr in enumerate(arrays):
            y_values = np.array(arr)
            kde = gaussian_kde(y_values)
            y_range = np.linspace(0, np.max(y_values), 100)
            pdf_kde = kde.evaluate(y_range)
            
            if k == idx:
                axes[1, idx].plot(y_range, pdf_kde, kde_colors[k] + '-', lw=2, label=kde_labels[k])
            else:
                axes[1, idx].plot(y_range, pdf_kde, kde_colors[k] + '-', lw=2, alpha=0.3, label=kde_labels[k])
        
        axes[1, idx].axvline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
        axes[1, idx].legend(fontsize='small')

        axes[1, idx].set_xlabel('distance from ball center')
        axes[1, idx].set_ylabel('value of probability mass function')
        
        axes[1, idx].set_title(kde_title[idx])
        # Add metric description to bottom right plot
        if metric_name is not None:
            axes[1, n-1].set_title(f"{kde_title[idx]}\n{metric_name} {metric_value:.4f}")
        # else:
        #     axes[1, n-1].set_title(kde_title[idx])
        
        # axes[1, n-1].set_title('distance from ball center')

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_path}/epoch_{save_name}.png', dpi=300)
    
    # if save:
    #     plt.savefig(f'{save_path}/epoch_{save_name}.eps', format="eps", dpi=300)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

def plot_nodes_color_kde_precentil_binary_new(arrays, lables_arr, save_path, title='Train/Validation/Test 1-probabilities', kde_title='KDE plot',
                                      percentile=99, save=False, save_name='_',
                                      alpha_0=0.05, alpha_1=0.8, size_0=0.1, size_1=1.5,
                                      array_x=None, metric_name=None, metric_value=0):
    
    
    # Set up a gridspec with two columns and one row
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    fig = plt.figure(figsize=(16, 8))
    
    ax = fig.add_subplot(gs[0])
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    
    cmap = plt.get_cmap('viridis')
    viridis_dark_blue = cmap(0)
    viridis_yellow = cmap(0.98)
    
    # y_max = max([np.max(array) for array in arrays])
    y_max = 1.5
    kde_colors = ['r', 'b', 'g']
    kde_labels = ['kde train', 'kde validation', 'kde test']
    
    threshold_99 = None
    
    caption_labels = ['Normal', 'Attack']
    colors = [viridis_dark_blue, viridis_yellow]
    
    for idx, (array, label) in enumerate(zip(arrays, lables_arr)):
        
        labels = [int(x) for x in label]
        sizes = [size_0 if label == 0 else size_1 for label in labels]
        alpha = [alpha_0 if label == 0 else alpha_1 for label in labels]
        
        
        if array_x is not None:
            ax.scatter(array_x[idx], array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
        else:
            ax.scatter(range(len(array)), array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)

        if idx == 0:
            threshold_99 = np.percentile(array, percentile)
        # ax.axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
        
        if idx != 0:
            min_x_value = np.min(array_x[idx])
            ax.axvline(min_x_value, color='gray', linestyle='--')
    
    
        # means = [np.mean([array[i] for i in range(len(array)) if labels[i] == j]) for j in range(2)]
        # classes = ['Normal mean', 'Attack mean']
        
        # for j in range(0, 2):
        #     ax.axhline(means[j], color=colors[j], linestyle='--', label=classes[j])
    ax.axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
    
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    new_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=caption_labels[i]) for i in range(len(caption_labels))]
    ax.legend(handles=legend_elements + new_elements, labels=legend_labels + caption_labels, fontsize='small')
    
    ax.set_title(title)
    ax.set_ylim(top=y_max)
    ax.set_xlabel('time, ms')
    ax.set_ylabel('1-probability ')
    
    
    # Create a new axis with a shared x-axis
    # ax2 = ax.twinx()
    ax2 = fig.add_subplot(gs[1])
    
    # Loop to plot the rotated KDE data
    for k, arr in enumerate(arrays):
        y_values = np.array(arr)
        kde = gaussian_kde(y_values)
        y_range = np.linspace(0, np.max(y_values), 100)
        pdf_kde = kde.evaluate(y_range)
        
        if k == 2:
            ax2.plot(pdf_kde, y_range, kde_colors[k] + '-', lw=2, label=kde_labels[k])
        else:
            ax2.plot(pdf_kde, y_range, kde_colors[k] + '-', lw=2, alpha=0.3, label=kde_labels[k])

    ax2.axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
    ax2.set_ylim(top=y_max)
    
    # Move the legend of ax2 to the right
    ax2.legend(fontsize='small', bbox_to_anchor=(1.15, 1), loc='upper right')
    
    # Set axis labels for ax2
    # ax2.set_xlabel('value of probability mass function')
    ax2.set_xlabel('value of probability mass function')
        
    ax2.set_title(kde_title)
    # Add metric description to bottom right plot
    if metric_name is not None:
        ax2.set_title(f"{kde_title}\n{metric_name} {metric_value:.4f}")
    

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{save_path}/epoch_{save_name}.png', dpi=300)
    
    # if save:
    #     plt.savefig(f'{save_path}/epoch_{save_name}.eps', format="eps", dpi=300)
    plt.show()




def plot_nodes_color_kde_precentil_binary_new_32(arrays, lables_arr, save_path, title='Train/Validation/Test distances', kde_title='KDE plot',
                                      percentile=99.5, save=False, save_name='_',
                                      alpha_0=0.05, alpha_1=0.8, size_0=0.1, size_1=1.5,
                                      array_x=None, metric_name=None, metric_value=0):
    
    
    # Set up a gridspec with two columns and one row
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    fig = plt.figure(figsize=(16, 8))
    
    ax = fig.add_subplot(gs[0])
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    
    cmap = plt.get_cmap('viridis')
    viridis_dark_blue = cmap(0)
    viridis_yellow = cmap(0.98)
    
    # y_max = max([np.max(array) for array in arrays])
    y_max = 11
    kde_colors = ['r', 'b', 'g']
    kde_labels = ['kde train', 'kde validation', 'kde test']
    
    threshold_99 = None
    
    caption_labels = ['Normal', 'Attack', '(32, *)', '(*, 32)', '(11, *)', '(*, 11)']
    colors = [viridis_dark_blue, viridis_yellow, 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
    
    for idx, (array, label) in enumerate(zip(arrays, lables_arr)):
        
        labels = [int(x) for x in label]
        sizes = [size_0 if label == 0 else size_1 for label in labels]
        alpha = [alpha_0 if label == 0 else alpha_1 for label in labels]
        
        
        if array_x is not None:
            ax.scatter(array_x[idx], array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)
        else:
            ax.scatter(range(len(array)), array, c=[colors[label] for label in labels], alpha=alpha, s=sizes)

        if idx == 0:
            threshold_99 = np.percentile(array, percentile)
        # ax.axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
        
        if idx != 0:
            min_x_value = np.min(array_x[idx])
            ax.axvline(min_x_value, color='gray', linestyle='--')
    
    
        # means = [np.mean([array[i] for i in range(len(array)) if labels[i] == j]) for j in range(2)]
        # classes = ['Normal mean', 'Attack mean']
        
        # for j in range(0, 2):
        #     ax.axhline(means[j], color=colors[j], linestyle='--', label=classes[j])
    ax.axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
    
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    new_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=caption_labels[i]) for i in range(len(caption_labels))]
    ax.legend(handles=legend_elements + new_elements, labels=legend_labels + caption_labels, fontsize='small')
    
    ax.set_title(title)
    ax.set_ylim(top=y_max)
    ax.set_xlabel('time, ms')
    ax.set_ylabel('distance from ball center')
    
    
    # Create a new axis with a shared x-axis
    # ax2 = ax.twinx()
    ax2 = fig.add_subplot(gs[1])
    
    # Loop to plot the rotated KDE data
    for k, arr in enumerate(arrays):
        y_values = np.array(arr)
        kde = gaussian_kde(y_values)
        y_range = np.linspace(0, np.max(y_values), 100)
        pdf_kde = kde.evaluate(y_range)
        
        if k == 2:
            ax2.plot(pdf_kde, y_range, kde_colors[k] + '-', lw=2, label=kde_labels[k])
        else:
            ax2.plot(pdf_kde, y_range, kde_colors[k] + '-', lw=2, alpha=0.3, label=kde_labels[k])

    ax2.axhline(threshold_99, color='red', linewidth=2, linestyle='-', label=f'{percentile}th percentile')
    ax2.set_ylim(top=y_max)
    
    # Move the legend of ax2 to the right
    ax2.legend(fontsize='small', bbox_to_anchor=(1.15, 1), loc='upper right')
    
    # Set axis labels for ax2
    # ax2.set_xlabel('value of probability mass function')
    ax2.set_xlabel('value of probability mass function')
        
    ax2.set_title(kde_title)
    # Add metric description to bottom right plot
    if metric_name is not None:
        ax2.set_title(f"{kde_title}\n{metric_name} {metric_value:.4f}")
    

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{save_path}/epoch_{save_name}.png', dpi=300)
    
    # if save:
    #     plt.savefig(f'{save_path}/epoch_{save_name}.eps', format="eps", dpi=300)
    plt.show()






def scatter_plot_histogram(data, labels, num_bins=5, offset=0.1, title='Scatter Plot Histogram',
                           size=0.3, alpha=0.8, save_path=None):
    # Convert lists to NumPy arrays if necessary
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)

    # Calculate the histogram
    hist, bin_edges = np.histogram(data, bins=num_bins)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Create a color map
    cmap = plt.cm.get_cmap('cividis', len(unique_labels))

    # Create the scatter plot histogram
    fig, ax = plt.subplots()
    for i in range(num_bins):
        mask = np.logical_and(data >= bin_edges[i], data < bin_edges[i + 1])
        bin_data = data[mask]
        bin_labels = labels[mask]
        y_positions = np.arange(len(bin_data)) + offset
        for j, label in enumerate(unique_labels):
            label_indices = np.where(bin_labels == label)[0]
            ax.scatter(bin_data[label_indices], y_positions[label_indices], alpha=alpha, s=size, c=[cmap(j)])

    # Customize the plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(title)

    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Label {label}', markerfacecolor=cmap(j), markersize=8) for j, label in enumerate(unique_labels)]
    ax.legend(handles=legend_elements)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()



def histogram_with_kde(data, labels, num_bins=5, alpha=0.8, kde_alpha=0.6, title='Histogram with KDE',
                       use_labels_for_kde=True, save_path=None):
    # Lazy import seaborn to keep it optional for the canonical code path
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "seaborn is required for histogram_with_kde(). Install via `pip install seaborn`"
        ) from e
    # Convert lists to NumPy arrays if necessary
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Create a color map
    # 'viridis'
    # 'plasma'
    # 'inferno'
    # 'magma'
    # 'cividis'
    cmap = plt.cm.get_cmap('cividis', len(unique_labels) + 1)  # Add 1 for the single-class KDE color

    # Create the histogram with label coloring
    fig, ax = plt.subplots()
    n, bins, _ = ax.hist(data, bins=num_bins, alpha=0)

    # Create histograms for each label
    for j, label in enumerate(unique_labels):
        label_data = data[np.where(labels == label)]
        ax.hist(label_data, bins=bins, alpha=alpha, color=cmap(j), label=f'Label {label}')

    # Create a second axis for the KDE plot
    ax2 = ax.twinx()

    # Add KDE plot on top of the histogram
    if use_labels_for_kde:
        for j, label in enumerate(unique_labels):
            label_data = data[np.where(labels == label)]
            sns.kdeplot(label_data, ax=ax2, color=cmap(j), alpha=kde_alpha, lw=2)
    else:
        sns.kdeplot(data, ax=ax2, color=cmap(len(unique_labels)), alpha=kde_alpha, lw=2)

    # Customize the plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax2.set_ylabel('Density')
    ax.set_title(title)

    # Create a custom legend
    legend_elements = [Patch(facecolor=cmap(j), alpha=alpha, label=f'Label {label}') for j, label in enumerate(unique_labels)]
    ax.legend(handles=legend_elements)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()




















from sklearn.decomposition import PCA

def plot_pca(data, point_size=0.007):
    # Create a PCA object and fit it to the data
    pca = PCA(n_components=2)
    pca.fit(data)
    
    # Transform the data to 2D using the PCA object
    transformed_data = pca.transform(data)
    
    # Plot the transformed data
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=point_size)
    plt.show()


def plot_umap(data, n_neighbors=15, min_dist=0.1, point_size=0.1, random_state=42, lable=False):
    # Lazy import umap to keep it optional
    try:
        import umap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for plot_umap(). Install via `pip install umap-learn`"
        ) from e
    # Create a UMAP object and fit it to the data
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    transformed_data = umap_obj.fit_transform(data)
    
    # Plot the transformed data
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=point_size)
    plt.show()





def plot_umap_lable(data, n_neighbors=15, min_dist=0.1, point_size=0.1, random_state=42, label=False):
    # Lazy import umap to keep it optional
    try:
        import umap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for plot_umap_lable(). Install via `pip install umap-learn`"
        ) from e
    
    """
    # 1) data[:, -2] = 1 & data[:, -1] = 2 dark red for attack+dst
    # 2) data[:, -2] = 1 & data[:, -1] = 1 dark green for attack and src
    # 3) data[:, -2] = 0 & data[:, -1] = 2 dark blue for norm+dst in batch
    # 4) data[:, -2] = 0 & data[:, -1] = 1 magneta for norm+src om batch
    # 5) data[:, -2] = 0 & data[:, -1] = 0 light blue rest embeddings not involved in batch
    
    """
    
    
    # Create a UMAP object and fit it to the data
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    transformed_data = umap_obj.fit_transform(data[:, :-2]) # drop last 2 columns (labels)

    # Compute the colors and sizes for each point based on the label and label count
    labels = data[:, -2]
    counts = data[:, -1]
    

    colors = np.where((labels==1) & (counts==2), 'red', 
              np.where((labels==1) & (counts==1), 'green', 
              np.where((labels==0) & (counts==2), 'darkblue',
              np.where((labels==0) & (counts==1), 'magenta', 'lightblue'))))
    sizes = np.where((labels==1) & (counts==2), point_size*40, 
             np.where((labels==1) & (counts==1), point_size*40, 
             np.where((labels==0) & (counts==2), point_size*20, 
             np.where((labels==0) & (counts==1), point_size*20, point_size))))


    # Plot the transformed data with the computed colors and sizes
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=sizes, c=colors)#,  cmap=plt.cm.get_cmap('viridis', 8))

    plt.show()

