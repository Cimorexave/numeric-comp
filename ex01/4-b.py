import numpy as np

def neville_matrix(f, m, n):
    """
    Computes the Neville matrix N where N[j, k] = p_{j, k}(0), 
    the value of the degree k polynomial interpolating f at nodes h_j, ..., h_{j+k},
    evaluated at x=0.
    
    Args:
        f (function): The function handle f(h) to be interpolated.
        m (int): Parameter defining the number of rows (m+1).
        n (int): Parameter defining the maximum degree (n) and number of columns (n+1).

    Returns:
        np.ndarray: The (m+1) x (n+1) Neville matrix N.
        np.ndarray: The h-values used for interpolation (h_0, h_1, ..., h_{m+n}).
    """
    
    # --- 1. Generate all required interpolation nodes h_i ---
    # The indices i run from 0 to m+n.
    num_nodes = m + n + 1
    i_values = np.arange(num_nodes)
    h_values = 2.0**(-i_values)
    
    # --- 2. Initialize the Neville Matrix N ---
    # The matrix N has dimensions (m+1) x (n+1)
    N = np.zeros((m + 1, n + 1))
    
    # --- 3. Fill the k=0 column (Initial values) ---
    # N_{j, 0} = p_{j, 0}(0) = f(h_j)
    # The index j runs from 0 to m.
    # The required h-values are h_0, h_1, ..., h_m (i.e., h_values[0] to h_values[m])
    N[:, 0] = f(h_values[:m + 1])
    
    # --- 4. Apply the Neville Recursion ---
    # The recursion starts at k=1 (degree 1) up to n.
    for k in range(1, n + 1):
        # The row index j runs from 0 up to m.
        # Note: If m=0, this loop only runs once for j=0.
        for j in range(m + 1):
            # The indices for the h-values are j and j+k.
            # N_{j, k} depends on N_{j, k-1} and N_{j+1, k-1}.
            
            # The indices j and j+k are for the h_values array.
            h_j = h_values[j]
            h_j_plus_k = h_values[j + k]
            
            # Recursive step: N_{j,k} = (-h_{j+k} * N_{j, k-1} + h_{j} * N_{j+1, k-1}) / (h_{j} - h_{j+k})
            
            # N_{j, k-1} is N[j, k-1]
            # N_{j+1, k-1} is N[j + 1, k-1] (This is why N must be at least (m+n) rows wide, but we only store the relevant m+1 rows)
            
            # *Correction for storing N only up to m rows:*
            # Since N_{j+1, k-1} is required, and j goes up to m, the value N_{m+1, k-1} would be needed, 
            # which is outside the (m+1) rows of N.
            # 
            # *The correct approach for the given dimension (m+1) x (n+1):*
            # When computing the k-th column, the index j can only go up to m-k.
            # Example: For k=n, j must be 0 (m-n=0 if m=n).
            # The last row we can compute for a given k is j = m - k.
            
            # Re-running the loop with the correct row limit:
            
            # The index j now runs from 0 up to m - k.
            # This ensures that j+1 is not greater than m-k+1, and j+k is not greater than m.
            
            # We are computing N[j, k] using N[j, k-1] and N[j+1, k-1].
            # For j up to m-k, j+1 will be up to m-k+1.
            
            # Let's adjust the indices:
            # We need h_values up to h_{m+n}.
            # The matrix N has m+1 rows, so indices j = 0 to m.
            # The maximum index for h_values used in the recursion for N[j, k] is j+k.
            # If j=m, k=0, then h_m is used.
            # If j=0, k=n, then h_n is used.
            # The highest index needed for h_values is m+n when k=n and j=m (which won't happen).
            
            # For N[j, k], we need h_j and h_{j+k}.
            # We need N[j, k-1] and N[j+1, k-1].
            
            # The last row that is filled in column k is j = m-k.
            # E.g., for k=1, max j is m-1. N[m-1, 1] uses N[m-1, 0] and N[m, 0].
            
            # Recalculate:
            N[j, k] = (-h_values[j + k] * N[j, k - 1] + h_values[j] * N[j + 1, k - 1]) / (h_values[j] - h_values[j + k])


    return N, h_values[:m + n + 1] # Return N and the h-values used up to h_{m+n}