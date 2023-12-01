import matplotlib.pyplot as plt
import numpy as np

class live_plot:
    def __init__(self, title):
        # initialize plot
        self.f, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        self.f.suptitle(title)
        plt.ion()
        plt.show()

    def update(self, values, trend_line=True, max_line=True, goal_line=False, goal=None):
        # Clear the axis
        self.ax[0].cla()
        self.ax[1].cla()

        # Plot rewards
        self.ax[0].plot(values, label='score per run')
        if max_line:
            self.ax[0].axhline(max(values), c='red',ls='--', label='maxiumum')
        if goal_line:
            self.ax[0].axhline(goal, c='green',ls='--', label='goal')
        self.ax[0].set_xlabel('Episodes')
        self.ax[0].set_ylabel('Reward')
        x = range(len(values))

        # Calculate the trend line
        if trend_line:
            try:
                z = np.polyfit(x, values, 1)
                p = np.poly1d(z)
                self.ax[0].plot(x,p(x),"--", label='trend')
            except:
                print('Error plotting trend line')

        self.ax[0].legend()
        
        # Plot the histogram of results
        self.ax[1].hist(values[-50:])
        self.ax[1].set_xlabel('Scores per Last 50 Episodes')
        self.ax[1].set_ylabel('Frequency')
        if goal_line:
            self.ax[1].axvline(goal, c='green', label='goal')
            self.ax[1].legend()
        
        plt.draw()
        plt.pause(0.001)

    def save(self, filename):
        # Save plot
        plt.savefig(filename)

    def close(self):
        # Close plot
        plt.close()
