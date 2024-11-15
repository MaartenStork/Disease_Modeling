import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq

# Class for the SIR Model with Demography
class SIRWithDemography:
    """
    A class to represent the SIR model with demographic factors (birth and death rates).
    """
    def __init__(self, beta, gamma, mu, initial_conditions):
        """
        Constructs all the necessary attributes for the SIR model with demography.
        
        Parameters:
        -----------
        beta : float
            Transmission rate.
        gamma : float
            Recovery rate.
        mu : float
            Birth/death rate.
        initial_conditions : list
            Initial conditions for [S, I, R].
        """
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.initial_conditions = initial_conditions

    def sir_demography_equations(self, y, t):
        """
        Defines the system of differential equations for the SIR model with demography.
        
        Parameters:
        -----------
        y : list
            Current values of [S, I, R].
        t : float
            Current time step.
        
        Returns:
        --------
        list
            Derivatives [dS/dt, dI/dt, dR/dt].
        """
        S, I, R = y
        dSdt = self.mu - self.beta * S * I - self.mu * S
        dIdt = self.beta * S * I - self.gamma * I - self.mu * I
        dRdt = self.gamma * I - self.mu * R
        return [dSdt, dIdt, dRdt]

    def simulate(self, t):
        """
        Simulates the SIR model with demography over a given time period.
        
        Parameters:
        -----------
        t : ndarray
            Array of time points.
        """
        self.results = odeint(self.sir_demography_equations, self.initial_conditions, t)

    def plot_results(self, t):
        """
        Plots the time evolution of the Susceptible, Infected, and Recovered populations.
        
        Parameters:
        -----------
        t : ndarray
            Array of time points.
        """
        S, I, R = self.results.T
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, I, label='Infected')
        plt.plot(t, R, label='Recovered')
        plt.xlabel('Time')
        plt.ylabel('Proportion of Population')
        plt.title('SIR Model with Demography')
        plt.legend()
        plt.show()

    def plot_phase_portrait(self):
        """
        Plots the phase portrait of the Susceptible vs. Infected populations.
        """
        S, I, _ = self.results.T
        plt.figure(figsize=(10, 6))
        plt.plot(S, I, label='Phase Portrait')
        plt.xlabel('Susceptible')
        plt.ylabel('Infected')
        plt.title('Phase Portrait of SIR Model with Demography')
        plt.legend()
        plt.show()

# Example usage
t = np.linspace(0, 1000, 1000)

# SIR Model with Demography
sir_demo_model = SIRWithDemography(beta=0.3, gamma=0.1, mu=0.01, initial_conditions=[0.99, 0.01, 0])
sir_demo_model.simulate(t)
sir_demo_model.plot_results(t)
sir_demo_model.plot_phase_portrait()
