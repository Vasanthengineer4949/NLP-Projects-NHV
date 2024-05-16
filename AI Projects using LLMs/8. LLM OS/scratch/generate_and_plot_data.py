import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to generate sample data
def generate_data():
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)
    return x, y

# Function to plot data
def plot_data(x, y):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue')
    plt.title('Scatter Plot of Sample Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    st.pyplot(plt)
    st.markdown('### Scatter Plot')
    st.markdown('The plot above shows the relationship between X and Y with noise added to Y.')

# Main execution block
if __name__ == "__main__":
    x, y = generate_data()
    plot_data(x, y)