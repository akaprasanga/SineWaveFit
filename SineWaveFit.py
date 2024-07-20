import numpy as np
import matplotlib.pyplot as plt

class SineWaveFit():

    def __init__(self, x_data, y_data, learning_rate=0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,iterations = 10000, patience = 500) -> None:
        self.X = x_data
        self.Y = y_data
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = iterations
        self.patience = patience


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
                print(f"Early stopping at iteration {i}")
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

    sineObj = SineWaveFit(x_data=x_data, y_data=y_true)
    fitted_parms = sineObj.fit()
    # Plot the results
    plt.scatter(x_data, y_true, label='Actual Data')
    y_pred = fitted_parms['amplitude'] * np.sin(fitted_parms['frequency'] * x_data + fitted_parms['phase']) + fitted_parms['vertical_shift']
    plt.plot(x_data, y_pred, label='Fitted Sine Wave', color='red')
    plt.legend()
    plt.show()
