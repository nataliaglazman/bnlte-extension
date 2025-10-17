import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



# Differentiable gate function (hard sigmoid)
def hard_sigmoid(s):
    return torch.clamp(s, 0.0, 1.0)

# Sample stochastic gates using Binary Concrete distribution
class StochasticGate(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def forward(self):
        eps = 1e-6
        device = self.log_alpha.device
        u = torch.rand_like(self.log_alpha, dtype=torch.float32, device=device)
        u = torch.clamp(u, eps, 1 - eps)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha))
        return hard_sigmoid(s)


    def expected_gate(self):
        return torch.sigmoid(self.log_alpha)

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 8)

        self.g1 = StochasticGate(self.fc1.weight.shape)
        self.g2 = StochasticGate(self.fc2.weight.shape)
        self.g3 = StochasticGate(self.fc3.weight.shape)

        self.g1_sample = self.g2_sample = self.g3_sample = None

    def resample_gates(self):
        # call this at the beginning of every new batch/trajectory
        self.g1_sample = self.g1()
        self.g2_sample = self.g2()
        self.g3_sample = self.g3()


    # def forward(self, z, s):
    #     W1 = self.fc1.weight * self.g1_sample
    #     W2 = self.fc2.weight * self.g2_sample
    #     W3 = self.fc3.weight * self.g3_sample
    #     x = torch.cat([z, s], dim=-1)  # [batch, 13], s broadcasted if needed

    #     x = F.silu(torch.nn.functional.linear(x, W1, self.fc1.bias))
    #     x = F.silu(torch.nn.functional.linear(x, W2, self.fc2.bias))
    #     x = torch.nn.functional.linear(x, W3, self.fc3.bias)
    #     return x
    def forward(self, z, s):
        # Concatenate state and static features
        x = torch.cat([z, s], dim=-1)  # [batch, input_dim]
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x


    def pathreg_term(self):
        G1 = self.g1.expected_gate()
        G2 = self.g2.expected_gate()
        G3 = self.g3.expected_gate()
        Az = torch.matmul(torch.matmul(G3.abs(), G2.abs()), G1.abs())
        return Az.sum()  # ||Az||_1,1

    def weightreg_term(self):
        A = torch.matmul(torch.matmul(self.fc3.weight.abs(), self.fc2.weight.abs()), self.fc1.weight.abs())
        return A.sum()  # ||A||_1,1

    def return_adjacency_matrix(self):
        # W = self.fc4.weight @ self.fc3.weight @ self.fc2.weight @ self.fc1.weight
        W =  self.fc3.weight @ self.fc2.weight @ self.fc1.weight
        return W.detach().cpu().numpy()

class NeuralODE(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, z0, s, t):

        # x0 = x0.to(dtype=torch.float32, device='cpu')
        # t = t.to(dtype=torch.float32, device='cpu')

        # return odeint(self.func, x0, t, rtol=1e-3, atol=1e-4, method='dopri5', options={'dtype': torch.float32}
        # z0 = z0.to(dtype=torch.float32)
        # t = t.to(dtype=torch.float32)
        # def ode_wrapper(z, t):
        #     return self.func(z, s)  # Inject s
        # return odeint(ode_wrapper, z0, t, rtol=1e-3, atol=1e-4, method='dopri5', options={'dtype': torch.float32})
    
        z0 = z0.to(dtype=torch.float32)
        t = t.to(dtype=torch.float32)
        s = s.to(dtype=torch.float32)  # Ensure dtype
        
        # Wrapper: Note order (t, z) to match odeint's expectation
        ode_wrapper = lambda t, z: self.func(z, s)  # Time t (ignored if not used), state z, inject s
        out = odeint(ode_wrapper, z0, t, rtol=1e-3, atol=1e-4, method='dopri5')
        return out  # [len(t), batch, dim]
    
    def return_adjacency_matrix(self):
        return self.func.return_adjacency_matrix()

    def pathreg_loss(self):
        return self.func.pathreg_term()

    def weightreg_loss(self):
        return self.func.weightreg_term()
    

if __name__ == "__main__":
    # Load and preprocess data
    device = torch.device('cpu')
    data = np.load('full_time_series2.npy')
    static_vars = data[:, :, :5]
    dynamic_vars = data[:,:, 5:]
    print('data shape:', data.shape, 'static_vars shape:', static_vars.shape, 'dynamic_vars shape:', dynamic_vars.shape)

    static_vars = torch.tensor(static_vars, dtype=torch.float32)
    dynamic_vars = torch.tensor(dynamic_vars, dtype=torch.float32)
    static_vars = static_vars.to(device)
    dynamic_vars = dynamic_vars.to(device)

    # data = torch.tensor(data, dtype=torch.float32)
    # data = data.to(device)

    patients = []
    T_max, N, D = data.shape
    print('T_max:', T_max, 'N:', N, 'D:', D)

    for i in range(N):
        # patient_data = data[:, i, :]
        static_patient_data = static_vars[: , i, :]
        dynamic_patient_data = dynamic_vars[:, i, :]
        mask = ~torch.isnan(dynamic_patient_data).any(dim=1)
        y_i = dynamic_patient_data[mask]
        x_i = static_patient_data[mask]
        t_i = torch.arange(T_max, device=device)[mask].float()

        if len(t_i) > 1:  # skip patients with ≤ 1 time point
            # sort and remove duplicates
            t_i, idx = torch.sort(t_i)
            y_i = y_i[idx]

            dt = t_i[1:] - t_i[:-1]
            valid = dt > 1e-6
            t_i = torch.cat([t_i[:1], t_i[1:][valid]])
            y_i = torch.cat([y_i[:1], y_i[1:][valid]])

            patients.append({'t': t_i, 'y': y_i, 'x': x_i})

    patients = [p for p in patients if p['t'].shape[0] > 1]
    
    # train_data, test_data, train_t, test_t = train_test_split(data, t, test_size=0.2, random_state=42)
    func = NeuralODE(ODEFunc()).to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)

    for it in range(1000):
        func.func.resample_gates()  # Resample gates before forward pass
        total_loss = 0.0
        
        for number, patient in enumerate(patients):
            train_patient = patient['y'].to(device)
            static_patient = patient['x'].to(device)
            train_patient_t = patient['t'].to(device)
            # Use patient-specific time vector here
            x0 = train_patient[0].unsqueeze(0)  # initial condition
            pred_y = func(x0, static_patient[0].unsqueeze(0), train_patient_t).squeeze(1)  # [T, D]

            total_loss += F.mse_loss(pred_y, train_patient)

        total_loss /= len(patients)

        optimizer.zero_grad()
        path_reg = func.pathreg_loss()
        weight_reg = func.weightreg_loss()
        total_loss.backward()
        optimizer.step()

        # if it % 10 == 0:
        #     # Evaluate on all patients using their observed times
        #     test_loss_total = 0.0
        #     with torch.no_grad():
        #         for patient in patients:  # or separate test patients if you split
        #             test_patient = patient['y'].to(device)
        #             test_patient_t = patient['t'].to(device)
        #             test_static_patient = patient['x'].to(device)
        #             x0 = test_patient[0].unsqueeze(0)
        #             test_static_patient = test_static_patient[0].unsqueeze(0)
        #             pred_y = func(x0, test_static_patient, test_patient_t).squeeze(1)
        #             test_loss_total += F.mse_loss(pred_y, test_patient)
        #         test_loss_total /= len(patients)
        #     print(f"Iter {it}, Train Loss: {total_loss.item():.4f}, Test Loss: {test_loss_total.item():.4f}, Path Reg: {path_reg.item():.4f}, Weight Reg: {weight_reg.item():.4f}")
            
            # Plot trajectories for each dimension
            # T_i, D = test_patient.shape
            # fig, axs = plt.subplots(D, 1, figsize=(10, 3 * D), sharex=True)
            # for d in range(D):
            #     axs[d].plot(test_patient_t.cpu().numpy(), test_patient[:, d].cpu().numpy(), label='Actual', marker='o')
            #     axs[d].plot(test_patient_t.cpu().numpy(), pred_y[:, d].cpu().numpy(), label='Predicted', marker='x')
            #     axs[d].set_ylabel(f'Dimension {d+1}')
            #     axs[d].legend()
            # axs[-1].set_xlabel('Time')
            # plt.suptitle('Predicted vs Actual Trajectories for a Test Patient')
            # plt.tight_layout()
            # plt.savefig('trajectories.png')  # Save to file or plt.show() for interactive

        if it % 10 == 0:
            # Evaluate on all patients: Collect MSEs and trajectories for plotting
            patient_mses = []
            actuals_list = []  # List of [T, D] for all patients
            preds_list = []    # List of [T, D] for all patients
            patient_dim_mses = []  # List of lists for per-dimension MSEs
            with torch.no_grad():
                for patient in patients:
                    if len(patient['t']) < 2: continue
                    
                    test_patient = patient['y'].to(device)  # [T, 9]
                    static_patient = patient['x'].to(device)
                    t = patient['t'].to(device)
                    
                    z0 = test_patient[0].unsqueeze(0)
                    s = static_patient[0].unsqueeze(0)
                    pred_y = func(z0, s, t).squeeze(1)  # [T, 9]
                    
                
                    # Per-patient MSE (for boxplot)
                    # for d in range(test_patient.shape[-1]):
                    #     dim_mse = F.mse_loss(pred_y[:, d], test_patient[:, d]).item()
                    #     per_dim_mse.append(dim_mse)

                    dim_mses = []
                    for d in range(test_patient.shape[-1]):
                        dim_mse = F.mse_loss(pred_y[:, d], test_patient[:, d]).item()
                        dim_mses.append(dim_mse)

                    patient_mse = F.mse_loss(pred_y, test_patient).item()
                    patient_mses.append(patient_mse)
                    patient_dim_mses.append(dim_mses)
                    
                    # Store for potential other plots
                    actuals_list.append(test_patient.cpu().numpy())
                    preds_list.append(pred_y.cpu().numpy())
                
                patient_dim_mses = np.array(patient_dim_mses)  # [num_patients, D]
                avg_test_mse = np.mean(patient_mses) if patient_mses else 0.0
                avg_test_mse_per_dim = np.mean(patient_dim_mses, axis=0)
                print('Avg Test MSE per dim:', avg_test_mse_per_dim)
            
            print(f"Iter {it}, Train Loss: {total_loss.item():.4f}, Avg Test MSE: {avg_test_mse:.4f}, Path Reg: {path_reg.item():.4f}, Weight Reg: {weight_reg.item():.4f}")
            
            # --- New: Box-and-Whisker Plot of Per-Patient MSEs ---
            plt.figure(figsize=(8, 6))
            plt.boxplot(patient_dim_mses, vert=True, patch_artist=True, labels=[f'Dim {i}' for i in range(8)])
            plt.title('Distribution of Per-Patient MSE (Test Set)')
            plt.ylabel('Mean Squared Error')
            plt.ylim(0, 2)
            plt.grid(True, axis='y')
            # Optional: sns.boxplot(data=patient_mses) if using seaborn
            plt.tight_layout()
            plt.savefig('error_distribution.png')
            plt.close()
            print(f"Saved boxplot to error_distribution.png (Median MSE: {np.median(patient_mses):.4f}, Std: {np.std(patient_mses):.4f})")
            

            # if actuals_list:  # If any patients
            #     last_actual = actuals_list[-10]  # [T, D]
            #     last_pred = preds_list[-10]
            #     last_t = patients[-10]['t'].cpu().numpy()  # Assuming last patient
            #     T_i, D = last_actual.shape
                
            #     fig, axs = plt.subplots(D, 1, figsize=(10, 3 * D), sharex=True)
            #     if D == 1: axs = [axs]  # Handle single dim
            #     for d in range(D):
            #         axs[d].plot(last_t, last_actual[:, d], label='Actual', marker='o')
            #         axs[d].plot(last_t, last_pred[:, d], label='Predicted', marker='x')
            #         axs[d].set_ylabel(f'Dimension {d+1}')
            #         axs[d].legend()
            #     axs[-1].set_xlabel('Time')
            #     plt.suptitle('Predicted vs Actual Trajectories for a Sample Patient')
            #     plt.tight_layout()
            #     plt.savefig('sample_trajectories.png')
            #     plt.close()
            #     print("Saved sample trajectories to sample_trajectories.png")

            cols = [ 'Hippocampus', 'Amygdala', 'Temporal Lobe',
            'ABETA42', 'TAU', 'PTAU',
                'TOTAL13', 'MMSCORE']
            if actuals_list:
                # Find indices of the 5 patients with the longest time series
                patient_lengths = [len(p['t']) for p in patients]
                top5_idx = np.argsort(patient_lengths)[-10:]  # indices of 5 longest patients

                print("Top 5 patients by number of time points:", patient_lengths)
                print("Selected indices (longest):", top5_idx)

                # Plot trajectories for these 5 patients
                fig, axs = plt.subplots(10, 1, figsize=(10, 20 * 3), sharex=False)
                if len(top5_idx) == 1:
                    axs = [axs]
                
                color_map_16 = plt.get_cmap('tab20').colors  # 20 distinct colors

                for ax, idx in zip(axs, top5_idx):
                    actual = actuals_list[idx]  # [T, D]
                    pred = preds_list[idx]
                    t = patients[idx]['t'].cpu().numpy()
                    T_i, D = actual.shape

                    # Plot only first few dims if D>4 to keep it readable
                    dims_to_plot = range(min(D, 8))
                    for d in dims_to_plot:
                        ax.plot(t, actual[:, d], marker='o', label=f'Actual Dim {cols[d]}', color=color_map_16[d*2])
                        ax.plot(t, pred[:, d], marker='x', linestyle='--', label=f'Pred Dim {cols[d]}', color=color_map_16[d*2+1])

                    ax.set_title(f'Patient {idx} (T={T_i})')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Value')
                    ax.legend(fontsize=8, loc ='lower left')
                    ax.grid(True)

                plt.suptitle('Predicted vs Actual Trajectories — 5 Longest Patients', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig('top5_longest_patient_trajectories.png')
                plt.close()
                print("Saved top 5 longest patient trajectories to top5_longest_patient_trajectories.png")