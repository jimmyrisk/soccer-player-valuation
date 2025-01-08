import numpy as np
import warnings

def compute_q_pi(P: np.ndarray, alpha: np.ndarray, S = None, U = None, threshold = 1e-8):
    """
    Computes vectors q and pi based on absorption probabilities into state S.

    Parameters:
    - P (np.ndarray): Transition matrix of the Markov chain.
                       Assumes S is the second-to-last state and U is the last state.
    - alpha (np.ndarray): Initial distribution over transient states.
    - S (int): Index of the absorbing state S (default: second-to-last state).  Defaults to N - 2.
    - U (int): Index of the absorbing state U (default: last state).  Defaults to N - 1.

    Returns:
    - q (np.ndarray): Vector where q[j] is the probability that j was involved in a trajectory absorbed into S.
    - pi (np.ndarray): Normalized vector computed from q.
    """
    # Total number of states
    N = P.shape[0]
    
    # Identify S and U indices
    if S is None:
        S = N - 2
    if U is None:
        U = N - 1
    
    # Define transient states (excluding S and U)
    transient_states = [state for state in range(N) if state not in [S, U]]
    
    # Number of transient states
    num_transient = len(transient_states)
    
    # Extract Q and R for the normal case (only S and U absorbing)
    Q_normal = P[np.ix_(transient_states, transient_states)]
    R_normal = P[np.ix_(transient_states, [S, U])]
    
    # Fundamental Matrix for normal case
    I_normal = np.eye(Q_normal.shape[0])
    N_normal = np.linalg.inv(I_normal - Q_normal)
    
    # Absorption Probabilities for normal case
    B_normal = N_normal @ R_normal
    prob_S_normal = alpha @ B_normal[:, 0]  # Probability of absorption into S
    
    # Initialize q vector
    q = np.zeros(num_transient)
    
    # Iterate over each transient state j
    for idx, j in enumerate(transient_states):
        # Make a copy of P to modify
        P_modified = P.copy()
        
        # Make state j absorbing
        P_modified[j, :] = 0.0
        P_modified[j, j] = 1.0
        
        # Define new absorbing states: S, U, and j
        absorbing_states_j = [S, U, j]
        
        # Define new transient states: all except S, U, and j
        transient_states_j = [state for state in transient_states if state != j]
        
        # Handle case where j is already absorbing (to avoid empty transient_states_j)
        if not transient_states_j:
            # If j is the only transient state, probability of S absorption remains the same
            prob_S_j = prob_S_normal
        else:
            # Extract Q and R for the modified case
            Q_j = P_modified[np.ix_(transient_states_j, transient_states_j)]
            R_j = P_modified[np.ix_(transient_states_j, absorbing_states_j)]
            
            # Fundamental Matrix for modified case
            I_j = np.eye(Q_j.shape[0])
            try:
                N_j = np.linalg.inv(I_j - Q_j)
            except np.linalg.LinAlgError:
                raise ValueError(f"Matrix (I - Q) is singular for state j={j} and cannot be inverted.")
            
            # Absorption Probabilities for modified case
            B_j = N_j @ R_j
            
            # Adjust the initial distribution by removing the probability of j
            alpha_transient_j = alpha.copy()
            alpha_transient_j = np.delete(alpha_transient_j, transient_states.index(j))
            
            # Compute probability of absorption into S
            prob_S_j = alpha_transient_j @ B_j[:, absorbing_states_j.index(S)]
        
        # Compute q_j value
        q_j = 1 - (prob_S_j / prob_S_normal) if prob_S_normal != 0 else 0
        # Set non-negative q_j value based on threshold.  Happens due to machine precision.
        if q_j < 0:
            if q_j < -threshold:
                warnings.warn(f"q_j value {q_j} for state {j} was below -{threshold}. Setting to 0.")
            q_j = 0

        q[idx] = q_j
    
    # Compute pi by normalizing q
    sum_q = np.sum(q)
    pi = q / sum_q if sum_q != 0 else np.zeros_like(q)
    
    return q, pi


def preprocess_passing_matrix(P, weight_successful=5/6, weight_unsuccessful=1/6, S_successful=None, S_unsuccessful=None, U=None):
    # Get dimensions 
    N = P.shape[0] - 3
    
    if S_successful is None:
        S_successful = N+1
    if S_unsuccessful is None:
        S_unsuccessful = N+2 
    if U is None:
        U = N+3
        
    # Create new matrix with N+2 rows/cols
    P_new = np.zeros((N+2, N+2))
    
    # Copy over player-to-player passing
    P_new[:N, :N] = P[:N, :N]
    
    # Combine successful/unsuccessful into single Success state
    P_new[:N, N] = weight_successful * P[:N, S_successful-1] + weight_unsuccessful * P[:N, S_unsuccessful-1]
    P_new[N, :N] = weight_successful * P[S_successful-1, :N] + weight_unsuccessful * P[S_unsuccessful-1, :N]
    
    # Add unsuccessful column
    P_new[:N, N+1] = P[:N, U-1]
    P_new[N+1, :N] = P[U-1, :N]
    
    # Normalize rows
    row_sums = P_new.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

    P_new = P_new / row_sums[:, np.newaxis]

    # Make S and U absorbing states
    P_new[N, N] = 1
    P_new[N, N+1] = 0
    P_new[N+1, N] = 0
    P_new[N+1, N+1] = 1

    return P_new
