import psutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

def monitor_cpu_usage(duration_seconds):
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration_seconds)

    timestamps = []
    cpu_percentages = []

    while datetime.now() < end_time:
        timestamps.append(datetime.now())
        cpu_percentages.append(psutil.cpu_percentages())

    return timestamps, cpu_percentages

def save_plot(timestamps, cpu_percentages, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, cpu_percentages, linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Load average (1 min)')
    plt.title('CPU Usage Over Time (modelling 5 images)')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    duration_seconds = 300  # 300s = 5 mins
    output_file = './plots/cpu_usage_plot_load.png'

    timestamps, cpu_percentages = monitor_cpu_usage(duration_seconds)
    save_plot(timestamps, cpu_percentages, output_file)
    print(f"CPU usage plot saved to {output_file}")
