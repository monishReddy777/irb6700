import numpy as np
from scipy.optimize import minimize
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the kinematic parameters of the robot
link_lengths = [5.25, 10.75, 2, 11.42, 1.70, 2.00]  # Lengths of each link
joint_limits = [(-180, 180), (-65, 85), (-180, 70), (-300, 300), (-120, 120), (-300, 300)]  # Joint limits

# Helper function to create a transformation matrix
def transformation_matrix(a, alpha, d, theta):
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])

# Forward kinematics function
def forward_kinematics(joint_angles):
    # DH parameters: (a, alpha, d, theta)
    dh_params = [
        (0,-90, link_lengths[0], joint_angles[0]),
        (link_lengths[1], 0, 0, joint_angles[1]),
        (link_lengths[2], -90, 0, joint_angles[2]),
        (0, -90, link_lengths[3], joint_angles[3]),
        (0, 90, 0, joint_angles[4]),
        (0, 0, link_lengths[4] + link_lengths[5], joint_angles[5])
    ]
    
    # Initial transformation matrix
    T = np.eye(4)
    
    # Initialize positions list with the base position
    positions = [T[:3, 3]]
    
    # Apply each DH transformation and store positions
    for param in dh_params:
        T = T @ transformation_matrix(*param)
        positions.append(T[:3, 3])
    
    return np.array(positions)

# Objective function
def objective_function(joint_angles, end_effector_desired):
    end_effector_actual = forward_kinematics(joint_angles)[-1]
    error = end_effector_actual - end_effector_desired
    return np.sum(error**2)

# Initial guess for joint angles
initial_guesses = [
    np.zeros(6),
    np.array([45, 0, 0, 0, 0, 0]),
    np.array([0, 45, 0, 0, 0, 0]),
    np.array([0, 0, 45, 0, 0, 0]),
    np.array([0, 0, 0, 45, 0, 0]),
    np.array([0, 0, 0, 0, 45, 0]),
    np.array([0, 0, 0, 0, 0, 45])
]

# Define bounds for the joint angles
bounds = joint_limits

# Streamlit app title
st.title("Robot Kinematics Optimization")

# Streamlit input widgets
x_desired = st.number_input("Enter the desired x-coordinate of the end-effector:", value=2.0)
y_desired = st.number_input("Enter the desired y-coordinate of the end-effector:", value=2.0)
z_desired = st.number_input("Enter the desired z-coordinate of the end-effector:", value=2.0)

end_effector_desired = np.array([x_desired, y_desired, z_desired])

# Store optimized joint angles and end-effector positions
optimized_results = []

# Run optimization when the button is clicked
if st.button("Optimize"):
    # Run optimization for each initial guess
    for initial_guess in initial_guesses:
        result = minimize(objective_function, initial_guess, args=(end_effector_desired,), method='SLSQP', bounds=bounds)
        optimized_joint_angles = result.x
        optimized_end_effector_position = forward_kinematics(optimized_joint_angles)[-1]
        optimized_results.append((optimized_joint_angles, optimized_end_effector_position))

    # Find the best result based on the minimum objective function value
    best_result = min(optimized_results, key=lambda x: objective_function(x[0], end_effector_desired))
    best_joint_angles, best_end_effector_position = best_result

    # Display results
    st.write("Best Optimized Joint Angles:", best_joint_angles)
    st.write("Best Optimized End-Effector Position:", best_end_effector_position)
    st.write("Optimization Results:")
    for idx, (joint_angles, end_effector_position) in enumerate(optimized_results):
        st.write(f"Initial Guess {idx+1}: Joint Angles = {joint_angles}, End-Effector Position = {end_effector_position}")

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the robot structure for the best result
    positions = forward_kinematics(best_joint_angles)
    for i in range(len(positions)-1):
        ax.plot([positions[i][0], positions[i+1][0]], 
                [positions[i][1], positions[i+1][1]], 
                [positions[i][2], positions[i+1][2]], 'b-')

    # Plot the desired end-effector position
    ax.scatter(end_effector_desired[0], end_effector_desired[1], end_effector_desired[2], color='r', label='Desired Position')

    # Plot the optimized end-effector positions
    optimized_positions = [result[1] for result in optimized_results]
    optimized_positions = np.array(optimized_positions)
    ax.scatter(optimized_positions[:, 0], optimized_positions[:, 1], optimized_positions[:, 2], color='g', label='Optimized Positions')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show plot
    st.pyplot(fig)
     
    ax = fig.add_subplot(111, projection='3d')

    # Plot the robot structure for the best result
    positions = forward_kinematics(best_joint_angles)
    for i in range(len(positions)-1):
        ax.plot([positions[i][0], positions[i+1][0]], 
                [positions[i][1], positions[i+1][1]], 
                [positions[i][2], positions[i+1][2]], 'b-')

    # Plot the desired end-effector position
    ax.scatter(end_effector_desired[0], end_effector_desired[1], end_effector_desired[2], color='r', label='Desired Position')

    # Plot the optimized end-effector positions
    optimized_positions = [result[1] for result in optimized_results]
    optimized_positions = np.array(optimized_positions)
    ax.scatter(optimized_positions[:, 0], optimized_positions[:, 1], optimized_positions[:, 2], color='g', label='Optimized Positions')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show plot
    st.pyplot(fig)

    # Optimization progress and convergence plot
    fig, ax = plt.subplots()
    for idx, (joint_angles, _) in enumerate(optimized_results):
        error_values = []
        def callback(xk):
            error_values.append(objective_function(xk, end_effector_desired))
        minimize(objective_function, initial_guesses[idx], args=(end_effector_desired,), method='SLSQP', bounds=bounds, callback=callback)
        ax.plot(error_values, label=f'Initial Guess {idx+1}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Function Value')
    ax.legend()
    st.pyplot(fig)

    # Visualize joint angles
    fig, ax = plt.subplots()
    ax.plot(best_joint_angles, 'bo-', label='Best Joint Angles')
    ax.set_xlabel('Joint')
    ax.set_ylabel('Angle (degrees)')
    ax.legend()
    st.pyplot(fig)

    # Save and load configurations
    if st.button("Save Configuration"):
        np.save("best_joint_angles.npy", best_joint_angles)
        st.write("Configuration saved.")
    if st.button("Load Configuration"):
        loaded_joint_angles = np.load("best_joint_angles.npy")
        loaded_end_effector_position = forward_kinematics(loaded_joint_angles)[-1]
        st.write("Loaded Joint Angles:", loaded_joint_angles)
        st.write("Loaded End-Effector Position:", loaded_end_effector_position)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        positions = forward_kinematics(loaded_joint_angles)
        for i in range(len(positions)-1):
            ax.plot([positions[i][0], positions[i+1][0]], 
                    [positions[i][1], positions[i+1][1]], 
                    [positions[i][2], positions[i+1][2]], 'b-')
        ax.scatter(end_effector_desired[0], end_effector_desired[1], end_effector_desired[2], color='r', label='Desired Position')
        ax.scatter(loaded_end_effector_position[0], loaded_end_effector_position[1], loaded_end_effector_position[2], color='g', label='Loaded Position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        st.pyplot(fig)

