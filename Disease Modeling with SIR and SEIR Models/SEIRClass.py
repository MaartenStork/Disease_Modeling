import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq

# Class for the SEIR Model
class SEIRModel:
    """
    A class to represent the SEIR (Susceptible-Exposed-Infected-Recovered) epidemiological model.
    
    Attributes:
    -----------
    beta : float
        The transmission rate of the disease.
    gamma : float
        The recovery rate of the disease.
    sigma : float
        The rate at which individuals move from the exposed to the infected class.
    initial_conditions : list
        Initial values for the Susceptible, Exposed, Infected, and Recovered populations.
    """
    def __init__(self, beta, gamma, sigma, initial_conditions):
        """
        Constructs all the necessary attributes for the SEIR model.
        
        Parameters:
        -----------
        beta : float
            Transmission rate.
        gamma : float
            Recovery rate.
        sigma : float
            Rate of progression from exposed to infected.
        initial_conditions : list
            Initial conditions for [S, E, I, R].
        """
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.initial_conditions = initial_conditions

    def seir_equations(self, y, t):
        """
        Defines the system of differential equations for the SEIR model.
        
        Parameters:
        -----------
        y : list
            Current values of [S, E, I, R].
        t : float
            Current time step.
        
        Returns:
        --------
        list
            Derivatives [dS/dt, dE/dt, dI/dt, dR/dt].
        """
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def simulate(self, t):
        """
        Simulates the SEIR model over a given time period.
        
        Parameters:
        -----------
        t : ndarray
            Array of time points.
        """
        self.results = odeint(self.seir_equations, self.initial_conditions, t)

    def plot_results(self, t):
        """
        Plots the time evolution of the Susceptible, Exposed, Infected, and Recovered populations.
        
        Parameters:
        -----------
        t : ndarray
            Array of time points.
        """
        S, E, I, R = self.results.T
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, E, label='Exposed')
        plt.plot(t, I, label='Infected')
        plt.plot(t, R, label='Recovered')
        plt.xlabel('Time')
        plt.ylabel('Proportion of Population')
        plt.title('SEIR Model')
        plt.legend()
        plt.show()

    def plot_phase_portrait(self):
        """
        Plots the phase portrait of the Susceptible vs. Infected populations.
        """
        S, I, _ = self.results.T[::10]  # Reduce points for better visualization
        plt.figure(figsize=(10, 6))
        plt.plot(S, I, label='Phase Portrait')
        plt.xlabel('Susceptible')
        plt.ylabel('Infected')
        plt.title('Phase Portrait of SEIR Model')
        plt.legend()
        plt.show()

    def plot_fourier_transform(self, t):
        """
        Plots the Fourier Transform of the Infected population over time.
        
        Parameters:
        -----------
        t : ndarray
            Array of time points.
        """
        I = self.results.T[2]
        N = len(t)
        yf = fft(I)
        xf = fftfreq(N, (t[1] - t[0]))[:N // 2]
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of Infected Population in SEIR Model')
        plt.show()

