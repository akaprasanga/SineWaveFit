import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class SineWaveFit():

    def __init__(self, x_data, y_data, learning_rate=0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,iterations = 10000, patience = 500, amp_init=0, freq_init=0, phase_init=0, vertical_shift_init=0) -> None:
        self.X = x_data
        self.Y = y_data
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = iterations
        self.patience = patience
        self.amp_init = amp_init
        self.freq_init = freq_init
        self.phase_init = phase_init
        self.vertical_shift_init = vertical_shift_init


    # Loss function: Mean Squared Error
    def loss_function(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    # Sine Wave defination
    def sine_function(self, xdata, amplitude, frequency, phase, vertical_shift):
        y = amplitude * np.sin(frequency * xdata + phase) + vertical_shift
        return y
    
    # Fitting Sine Wave
    def fit(self):
        # Initial parameters
        amp = np.random.randn()
        freq = np.random.randn()
        phase = np.random.randn()
        vertical_shift = np.random.randn()

                # Initial parameters
        # amp = self.amp_init
        # freq = self.freq_init
        # phase = self.phase_init
        # vertical_shift = self.vertical_shift_init

        # Initialize Adam optimizer variables
        mAmp, mFreq, mPhase, mVert = 0, 0, 0, 0
        vAmp, vFreq, vPhase, vVert = 0, 0, 0, 0
        t = 0

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0

        # Gradient descent with Adam optimizer and early stopping
        for i in range(self.iterations):
            t += 1
            
            # Predicted values
            y_pred = self.sine_function(xdata=self.X, amplitude=amp, frequency=freq, phase=phase, vertical_shift=vertical_shift)
            
            # Compute gradients
            dL_dAmp = np.mean(2 * (y_pred - self.Y) * np.sin(freq * self.X + phase))
            dL_dFreq = np.mean(2 * (y_pred - self.Y) * amp * self.X * np.cos(freq * self.X + phase))
            dL_dPhase = np.mean(2 * (y_pred - self.Y) * amp * np.cos(freq * self.X + phase))
            dL_dVert = np.mean(2 * (y_pred - self.Y))
            
            # Update biased first moment estimate
            mAmp = self.beta1 * mAmp + (1 - self.beta1) * dL_dAmp
            mFreq = self.beta1 * mFreq + (1 - self.beta1) * dL_dFreq
            mPhase = self.beta1 * mPhase + (1 - self.beta1) * dL_dPhase
            mVert = self.beta1 * mVert + (1 - self.beta1) * dL_dVert
            
            # Update biased second moment estimate
            vAmp = self.beta2 * vAmp + (1 - self.beta2) * (dL_dAmp ** 2)
            vFreq = self.beta2 * vFreq + (1 - self.beta2) * (dL_dFreq ** 2)
            vPhase = self.beta2 * vPhase + (1 - self.beta2) * (dL_dPhase ** 2)
            vVert = self.beta2 * vVert + (1 - self.beta2) * (dL_dVert ** 2)
            
            # Compute bias-corrected first moment estimate
            mAmp_hat = mAmp / (1 - self.beta1 ** t)
            mFreq_hat = mFreq / (1 - self.beta1 ** t)
            mPhase_hat = mPhase / (1 - self.beta1 ** t)
            mVert_hat = mVert / (1 - self.beta1 ** t)
            
            # Compute bias-corrected second moment estimate
            vAmp_hat = vAmp / (1 - self.beta2 ** t)
            vFreq_hat = vFreq / (1 - self.beta2 ** t)
            vPhase_hat = vPhase / (1 - self.beta2 ** t)
            vVert_hat = vVert / (1 - self.beta2 ** t)
            
            # Update parameters
            amp -= self.learning_rate * mAmp_hat / (np.sqrt(vAmp_hat) + self.epsilon)
            freq -= self.learning_rate * mFreq_hat / (np.sqrt(vFreq_hat) + self.epsilon)
            phase -= self.learning_rate * mPhase_hat / (np.sqrt(vPhase_hat) + self.epsilon)
            vertical_shift -= self.learning_rate * mVert_hat / (np.sqrt(vVert_hat) + self.epsilon)
            
            # Compute current loss
            loss = self.loss_function(y_pred, self.Y)
            
            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at iteration {i} with best loss {loss}")
                break
            
            # Print the loss every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
            
        return {'amplitude':amp, 'frequency':freq, 'phase':phase, 'vertical_shift':vertical_shift}

if __name__ == "__main__":
    # Generate synthetic data for a sine wave
    np.random.seed(0)
    x_data = np.linspace(0, 2 * np.pi, 100)
    y_true = 3 * np.sin(2 * x_data + 1) + 0.5 + np.random.normal(scale=1.5, size=x_data.shape)

    vertical_shift_initial = 0  # Initial guess for vertical shift D
    amp_initial = (np.max(y_true) - np.min(y_true)) / 2  # Initial guess for amplitude A

    # Rough initial guess for frequency f
    # Assuming the dataset has roughly one complete cycle
# Identify peaks to estimate the frequency
    peaks, _ = find_peaks(y_true)
    if len(peaks) > 1:
        peak_distances = np.diff(x_data[peaks])
        period_estimate = np.mean(peak_distances)
        freq_initial = 1 / period_estimate
    else:
        freq_initial = 1 / (2 * np.pi)  # Default to one complete cycle over the range

    phase_initial = 0  # Initial guess for phase shift C

    sineObj = SineWaveFit(x_data=x_data, y_data=y_true, amp_init=amp_initial, freq_init=freq_initial, phase_init=phase_initial, vertical_shift_init=vertical_shift_initial)
    fitted_parms = sineObj.fit()
    # Plot the results
    plt.scatter(x_data, y_true, label='Actual Data')
    y_pred = fitted_parms['amplitude'] * np.sin(fitted_parms['frequency'] * x_data + fitted_parms['phase']) + fitted_parms['vertical_shift']
    plt.plot(x_data, y_pred, label='Fitted Sine Wave', color='red')
    plt.legend()
    plt.show()
