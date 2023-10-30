import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data for the curve
#x = [i+1 for i in range(10)]
# for my own implementation
# y_300 = [0.8627804487179487, 0.887520032051282, 0.9017427884615384, 0.9103565705128205, 0.9181690705128205, 0.9190705128205128, 0.9250801282051282, 0.9271834935897436, 0.9320913461538461, 0.93359375]
# y_200 = [0.850761217948718, 0.8882211538461539, 0.9022435897435898, 0.9100560897435898, 0.9147636217948718, 0.9190705128205128, 0.9250801282051282, 0.9275841346153846, 0.9289863782051282, 0.9297876602564102]

x = [i+1 for i in range(20)]

# for the pytorch version
y_300 = [0.896, 0.9094, 0.9165, 0.9167, 0.9246, 0.9265, 0.9325, 0.9304, 0.9342, 0.9361, 0.9417, 0.9433, 0.9439, 0.9465, 0.949, 0.9498, 0.9522, 0.9542, 0.9543, 0.9556]
y_200 = [0.8972, 0.9107, 0.9158, 0.9209, 0.9273, 0.9297, 0.9318, 0.9364, 0.9384, 0.9401, 0.9424, 0.9455, 0.9474, 0.9496, 0.9505, 0.9519, 0.9539, 0.9535, 0.9567, 0.9576]
# Create the plot
plt.figure(figsize=(8, 6))  # Define the figure size (optional)
plt.plot(x, y_300, label='hidden dim = 300', color='r', linewidth=2)  
plt.plot(x, y_200, label='hidden dim = 200', color='b', linewidth=2)  


# normal = [0.8782, 0.8978, 0.9056, 0.9075, 0.9148, 0.9151, 0.9204, 0.9194, 0.9212, 0.925, 0.9262, 0.9282, 0.9281, 0.9317, 0.9332, 0.9333, 0.9355, 0.9355, 0.9369, 0.9381]
# zero = [0.1032, 0.1135, 0.3309, 0.5901, 0.7466, 0.823, 0.8599, 0.8814, 0.8934, 0.8997, 0.9042, 0.9072, 0.9105, 0.9128, 0.9136, 0.9163, 0.9179, 0.9179, 0.9213, 0.9224]
# uniform = [0.8244, 0.8693, 0.884, 0.8944, 0.8992, 0.9077, 0.9129, 0.9155, 0.9181, 0.922, 0.9237, 0.9261, 0.9288, 0.9297, 0.93, 0.9326, 0.9333, 0.9332, 0.9363, 0.9369]
# plt.figure(figsize=(8, 6))  # Define the figure size (optional)
# plt.plot(x, normal, label='normal init', color='r', linewidth=2)  
# plt.plot(x, zero, label='zero init', color='b', linewidth=2)  
# plt.plot(x, uniform, label='uniform init', color='g', linewidth=2)  


# Add labels and a legend
plt.xlabel('training epoch')
plt.ylabel('test accuracy')
#plt.title('self-implemented 2 layers NN with hidden dim = 300 or 200')
plt.title('pytorch 2 layers NN with hidden dim = 300 or 200')
#plt.title('performance with different initialization')
plt.legend()

# Show the plot
plt.grid(True)  # Add a grid (optional)
plt.show()
