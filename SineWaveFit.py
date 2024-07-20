import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a sine wave
np.random.seed(0)
x_data = np.linspace(0, 2 * np.pi, 100)
y_true = 3 * np.sin(2 * x_data + 1) + 0.5 + np.random.normal(scale=1.5, size=x_data.shape)

# Initial parameters
amp = np.random.randn()
freq = np.random.randn()
phase = np.random.randn()
vertical_shift = np.random.randn()

# Hyperparameters
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
iterations = 10000
patience = 500

# Initialize Adam optimizer variables
mAmp, mFreq, mPhase, mVert = 0, 0, 0, 0
vAmp, vFreq, vPhase, vVert = 0, 0, 0, 0
t = 0

# Early stopping variables
best_loss = float('inf')
patience_counter = 0

# Sine Wave defination
def sine_function(xdata, amplitude, frequency, phase, vertical_shift):
    y = amplitude * np.sin(frequency * xdata + phase) + vertical_shift
    return y

# Loss function: Mean Squared Error
def loss_function(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Gradient descent with Adam optimizer and early stopping
for i in range(iterations):
    t += 1
    
    # Predicted values
    y_pred = sine_function(xdata=x_data, amplitude=amp, frequency=freq, phase=phase, vertical_shift=vertical_shift)
    
    # Compute gradients
    dL_dAmp = np.mean(2 * (y_pred - y_true) * np.sin(freq * x_data + phase))
    dL_dFreq = np.mean(2 * (y_pred - y_true) * amp * x_data * np.cos(freq * x_data + phase))
    dL_dPhase = np.mean(2 * (y_pred - y_true) * amp * np.cos(freq * x_data + phase))
    dL_dVert = np.mean(2 * (y_pred - y_true))
    
    # Update biased first moment estimate
    mAmp = beta1 * mAmp + (1 - beta1) * dL_dAmp
    mFreq = beta1 * mFreq + (1 - beta1) * dL_dFreq
    mPhase = beta1 * mPhase + (1 - beta1) * dL_dPhase
    mVert = beta1 * mVert + (1 - beta1) * dL_dVert
    
    # Update biased second moment estimate
    vAmp = beta2 * vAmp + (1 - beta2) * (dL_dAmp ** 2)
    vFreq = beta2 * vFreq + (1 - beta2) * (dL_dFreq ** 2)
    vPhase = beta2 * vPhase + (1 - beta2) * (dL_dPhase ** 2)
    vVert = beta2 * vVert + (1 - beta2) * (dL_dVert ** 2)
    
    # Compute bias-corrected first moment estimate
    mAmp_hat = mAmp / (1 - beta1 ** t)
    mFreq_hat = mFreq / (1 - beta1 ** t)
    mPhase_hat = mPhase / (1 - beta1 ** t)
    mVert_hat = mVert / (1 - beta1 ** t)
    
    # Compute bias-corrected second moment estimate
    vAmp_hat = vAmp / (1 - beta2 ** t)
    vFreq_hat = vFreq / (1 - beta2 ** t)
    vPhase_hat = vPhase / (1 - beta2 ** t)
    vVert_hat = vVert / (1 - beta2 ** t)
    
    # Update parameters
    amp -= learning_rate * mAmp_hat / (np.sqrt(vAmp_hat) + epsilon)
    freq -= learning_rate * mFreq_hat / (np.sqrt(vFreq_hat) + epsilon)
    phase -= learning_rate * mPhase_hat / (np.sqrt(vPhase_hat) + epsilon)
    vertical_shift -= learning_rate * mVert_hat / (np.sqrt(vVert_hat) + epsilon)
    
    # Compute current loss
    loss = loss_function(y_pred, y_true)
    
    # Early stopping check
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at iteration {i}")
        break
    
    # Print the loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}")

# Plot the results
plt.scatter(x_data, y_true, label='Actual Data')
y_pred = amp * np.sin(freq * x_data + phase) + vertical_shift
plt.plot(x_data, y_pred, label='Fitted Sine Wave', color='red')
plt.legend()
plt.show()
