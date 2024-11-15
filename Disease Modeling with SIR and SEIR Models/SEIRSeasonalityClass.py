import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq

# Class for the SEIR Model with Seasonality
class SEIRModelWithSeasonality(SEIRModel):
    """
    A class to represent the SEIR model with seasonality.
    
    Attributes:
    -----------
    beta0 : float
        The average transmission rate over the year.
    beta1 : float
        The amplitude of seasonal variation in the transmission rate.
    gamma : float
        The recovery rate of the disease.
    sigma : float
        The rate at which individuals move from the exposed to the infected class.
    initial_conditions : list
        Initial values for the Susceptible, Exposed, Infected, and Recovered populations.
    """
    def __init__(self, beta0, beta1, gamma, sigma, initial_conditions):
        """
        Constructs all the necessary attributes for the SEIR model with seasonality.
        
        Parameters:
        -----------
        beta0 : float
            Average transmission rate.
        beta1 : float
            Amplitude of seasonal variation.
        gamma : float
            Recovery rate.
        sigma : float
            Rate of progression from exposed to infected.
        initial_conditions : list
            Initial conditions for [S, E, I, R].
        """
        super().__init__(beta0, gamma, sigma, initial_conditions)
        self.beta1 = beta1

    def seir_seasonal_equations(self, y, t):
        """
        Defines the system of differential equations for the SEIR model with seasonality.
        
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
        beta_t = self.beta * (1 + self.beta1 * np.cos(2 * np.pi * t / 365))
        dSdt = -beta_t * S * I
        dEdt = beta_t * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def simulate(self, t):
        """
        Simulates the SEIR model with seasonality over a given time period.
        
        Parameters:
        -----------
        t : ndarray
            Array of time points.
        """
        self.results = odeint(self.seir_seasonal_equations, self.initial_conditions, t)

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
        plt.title('SEIR Model with Seasonality')
        plt.legend()
        plt.show()


# Example usage
t = np.linspace(0, 1000, 1000)

# SEIR Model with Seasonality
seir_seasonal_model = SEIRModelWithSeasonality(beta0=0.3, beta1=0.2, gamma=0.1, sigma=0.2, initial_conditions=[0.99, 0.0, 0.01, 0])
seir_seasonal_model.simulate(t)
seir_seasonal_model.plot_results(t)
