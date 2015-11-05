import numpy as N
import mpi


def filtering(U, S, V, tolerance, dim_max):
    """Removes lowest modes in U, S and V.

    U, V      : numpy 2D arrays
    S         : numpy 1D array (represents a diagonal matrix)
    tolerance : tolerance for filtering modes
    dim_max   : maximum number of modes
    """
    total_sum = N.sum(S)
    if total_sum == 0.and S.size == 1:
        total_sum = 1.

    for i in range(S.shape[0]):
        dim = i+1
        if N.sum(S[:i+1]) / total_sum > tolerance: break
    dim = min(dim, dim_max)

    # copy ensures an array is not a slice of a bigger memory zone
    if U.shape[1] != dim: U = U[:, :dim].copy()
    if S.shape[0] != dim: S = S[:dim].copy()
    if V.shape[1] != dim: V = V[:, :dim].copy()
    return (U, S, V)


def update(U, S, V, mean_snapshot, snapshot):
    """Update a svd with one snapshot.

    U, S, V       : existing svd
    mean_snapshot : mean snapshot
    snapshot      : a snapshot

    See http://dx.doi.org/10.1016/j.laa.2005.07.021
    """
    if mean_snapshot is None:
        # start off with a mode that will be thrown away by filtering: 0. singular value
        mean_snapshot = snapshot
        U = N.zeros([snapshot.shape[0], 1])
        if mpi.myid == 0:
            U[0, 0] = 1.
        S = N.zeros([1])
        V = N.ones([1, 1])

    else:
        # backup and update mean snapshot
        mean_snapshot_copy = mean_snapshot.copy()
        s_nb = V.shape[0]
        mean_snapshot = (s_nb * mean_snapshot + snapshot) / (s_nb + 1)

        # move to pod origin and project the snapshot on the pod basis
        snapshot -= mean_snapshot_copy
        s_proj = N.dot(U.T, snapshot)

        mpi.Allreduce(sendbuf=s_proj.copy(), recvbuf=s_proj, op=mpi.sum)

        h = snapshot - N.dot(U, s_proj)
        h_norm = N.linalg.norm(h)

        h_norm *= h_norm
        h_norm = mpi.allreduce(h_norm, op=mpi.sum)
        h_norm = N.sqrt(h_norm)

        # St = |S   U^T s_proj|
        #      |0      norm(h)|
        S = N.column_stack([N.diag(S), s_proj])
        S = N.vstack([S, N.zeros_like(S[0])])
        S[-1, -1] = h_norm

        # Ut = |U  q/norm(q)|
        if h_norm == 0.: h_norm = 1. # fix for h = 0
        U = N.column_stack([U, h / h_norm])

        # Vt = |V  0|
        #      |0  1|
        V = N.vstack([V, N.zeros_like(V[0])])
        V = N.column_stack([V, N.zeros_like(V[:,0])])
        V[-1, -1] = 1.

        (Ud, S, Vd_T) = N.linalg.svd(S)
        V = N.dot(V, Vd_T.T)
        Un, S, V = downgrade(S, V)
        U = N.dot(U, N.dot(Ud, Un))

    return (U, S, V, mean_snapshot)


def downgrade(S0, Vt):
    """DOC"""
    v = N.average(Vt, 0)
    for row in Vt: row -= v
    (Q, R) = N.linalg.qr(Vt)
    R = (S0*R).T # R = S0[:,N.newaxis] * R.T
    (Urot, S, V) = N.linalg.svd(R)
    V = N.dot(Q, V.T)
    return (Urot, S, V)


# TODO: parallel static svd
# def pstasvd(Snapshot, tol, maxsize, mpi.myid = None, nimg = None):
# 
# ## This function computes a POD basis from a snapshot matrix through the use of
# ## a correlaton matrix, and using the statis POD algorithm described in the
# ## theoretical manual.
#     mpi.myid = mpi.myid or MPI.COMM_WORLD.Get_rank()
#     nimg = nimg or Snapshot.shape[1]
# 
#     Rloc = N.dot(transpose(Snapshot), Snapshot)
#     R = N.empty_like(Rloc)
#     MPI.COMM_WORLD.Reduce(Rloc, R, op=MPI.SUM, root=0)
#     index0 = 0
#     
#     if mpi.myid == 0:
#         R = reshape(R, (nimg, nimg))
#         (Eig0, V0) = eig(R)
#         for i in range(nimg):
#             Rloc[i, 0] = Eig0[i, i]
#         for i in range(nimg):
#             Eig0[i, i] = Rloc[nimg - i - 1, 0]
#         for i in range(nimg):
#             for j in range(nimg):
#                 Rloc[i, j] = V0[i, j]
#         for i in range(nimg):
#             V0[:, i] = Rloc[:, nimg - i - 1]
#             
#         index0 = findindex(Eig0, tol, maxsize)
#         
#     else:
#         Eig0 = zeros([nimg, nimg], Float)
#         V0 = zeros([nimg, nimg], Float)
#     
#     # eig returns F_CONTIGUOUS array and MPI do not transfer ordering
#     Eig0 = N.require(Eig0, requirements=['C_CONTIGUOUS'])
#     V0   = N.require(V0  , requirements=['C_CONTIGUOUS'])
#     MPI.COMM_WORLD.Bcast(Eig0, root=0)
#     MPI.COMM_WORLD.Bcast(V0  , root=0)
#     nmod = MPI.COMM_WORLD.bcast(index0, root=0)    
# 
#     S = Eig0[:nmod, :nmod]
#     V = V0[:, :nmod]
#     
#     return (S, V, nmod)


# def eig(A):
#     (D, V) = Heigenvectors(A)
#     n = len(D)
#     Eig = zeros([n, n], Float)
#     for i in range(n):
#         Eig[i, i] = sqrt(abs(D[i]))
#     V = transpose(V)
#     return (Eig, V)

# def Heigenvalues(a, UPLO='L'):
#     return linalg.eigvalsh(a,UPLO)

# def comp_phi(A, S, V):
#     (n, m) = shape(S)
#     invS = zeros([n, m], Float)
#     for i in range(n):
#         invS[i, i] = 1. / S[i, i]
#     U = N.dot(A, N.dot(V, invS))
#     return U
