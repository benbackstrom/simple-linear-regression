
from matplotlib import pyplot


def plot_model(title, weight, bias, feature_title, feature_data, label_title, label_data):
    # Labels for axes and plot
    pyplot.title(title)
    pyplot.xlabel(feature_title)
    pyplot.ylabel(label_title)

    # Plot model
    min_x = min(feature_data)
    max_x = max(feature_data)

    start_x = int(min_x - 1)
    start_y = int(bias + start_x * weight)
    end_x = int(max_x + 1)
    end_y = int(bias + end_x * weight)
    pyplot.plot([start_x, end_x], [start_y, end_y], c='r')

    # Plot data
    pyplot.scatter(feature_data, label_data)

    # Show the plot
    pyplot.show()


def plot_loss(epoch_data, root_mean_squared_error):
    pyplot.figure()

    # Labels for axes and plot
    pyplot.title('Loss')
    pyplot.xlabel('Epoch Number')
    pyplot.ylabel('Squared Error')

    # Plot loss
    pyplot.plot(epoch_data, root_mean_squared_error)
    pyplot.legend()
    pyplot.ylim([root_mean_squared_error.min() * 0.95, root_mean_squared_error.max()])

    # Show the plot
    pyplot.show()
