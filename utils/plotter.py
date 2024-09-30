import matplotlib.pyplot as plt

def init_plot(graph_window_size):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([0] * graph_window_size, 'g-', linewidth=2)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, graph_window_size)
    ax.set_title('Gaze Direction (Real-Time)')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Direction')
    fig.show()
    return fig, ax, line

def update_plot(fig, ax, line, direction_history):
    line.set_ydata(direction_history)
    ax.draw_artist(ax.patch)
    ax.draw_artist(line)
    fig.canvas.flush_events()
