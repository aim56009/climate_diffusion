from typing import Any
import torch

class Constraint():

    def __init__(self, u_0, t_0=0.0, stabilization_parameter=1e-2, constraint_fct="downscaling") -> None:
        """Initialize the constraint. 

        The constraint is implementend as 
        0 = self.function(u, t) - self.function(u_0, t_0)

        When calling an instance of Constraint with constraint(u,t) the full stabilization term

        stab = self.stabilization_parameter * torch.matmul(self.pseudoinverse(u,t), self.function(u,t))

        is returned. Input argument u is a Nx1xHxW tensor 

        Args:
            u_0: Initial conditions for the constraint: tensor 1x1xHxW. Used to compute the value of the constraint at u_0 and t_0
            t_0: Initial conditions for the constraint: tensor 1x1xHxW
        """
        self.function = ConstraintFunction.which_costraint(constraint_fct, u_0, t_0)
        self.pseudoinverse = ConstraintPseudoinverse.constraint_fct(constraint_fct, u_0, t_0)
        self.stabilization_parameter = stabilization_parameter

        self.use_at_inference = True
        self.use_at_training = False # not yet implemented

    def turn_off_at_inference(self):
        self.use_at_inference = False

    def turn_on_at_inference(self):
        self.use_at_inference = True

    def set_stabilization_parameter(self, gamma):
        self.stabilization_parameter = gamma 

    def __call__(self, u, t):
        N_batch = u.size(0)
        H = u.size(2)
        W = u.size(3)

        return self.stabilization_parameter * torch.matmul(self.pseudoinverse(u,t), self.function(u,t).unsqueeze(2)).squeeze(2).reshape((N_batch, 1, H, W)) # batched matrix multiply: (Nx(H*W)xm) * (mx1) = Nx(H*W) -> reshape -> N x 1 x H x W (as the input) (the unsqueeze is necessary for the correct batched mmul)

class ConstraintFunction():

    @classmethod
    def which_costraint(cls, constraint_fct, u_0, t_0):
        if constraint_fct == "mean":
           return MeanConstraintFunction(u_0, t_0)
        elif constraint_fct == "boundary": 
           return BoundaryConstraintFunction(u_0, t_0)
        elif constraint_fct == "downscaling":
            return DownscalingConstraintFunction(u_0, t_0)
        else:
            ValueError("Unknown Constraint Type")  
        
    def __call__(self, u, t):
        """Computes the constraint g(u,t)

        Args: 
            u: Input tensor Nx1xHxW 
            t: Float, time 

        The constraint is implementend as 
        0 = self.function(u, t) - self.function(u_0, t_0)
        """
        return self.function(u, t) - self.val_0

class ConstraintPseudoinverse():

    @classmethod
    def constraint_fct(cls, constraint_fct, u_0, t_0):
        if constraint_fct == "mean":
           return MeanConstraintPseudoinverse(u_0, t_0)
        elif constraint_fct == "boundary": 
           return BoundaryConstraintPseudoinverse(u_0, t_0)
        elif constraint_fct == "downscaling": 
            return DownscalingConstraintPseudoinverse(u_0, t_0)
        else:
            ValueError("Unknown Constraint Type")  
        
    def __call__(self, u, t):
        """Calls the Moore-Penrose Pseudoinverse of the Jacobian of the constraint 
        
        Args: 
            u: Input tensor Nx1xHxW 
            t: Float, time 

        Returns 
            G(u): Pseudoinverse N x (H*W) x m 
        """
        return torch.linalg.pinv(self.jacobian(u, t))

# ### CONSTRAINTS LIBRARY 

class MeanConstraintFunction(ConstraintFunction):

    def __init__(self, u_0, t_0=0.) -> None:
        self.val_0 = self.function(u_0, t_0)

    def function(self, u, t=0.):
        """Implements the mean constraint
        
        Args: 
            u: Tensor Nx1xHxW input argument
            t: Float time, ignored 

        Returns:
            mean as Nx1 tensor 
        """
        return u.mean(dim=(2,3))

    @staticmethod 
    def initialcondition(mu, u_0):
        """Helper function to prepare an 'initial condition' for the MeanConstraintFunction 
        
        Args: 
            mu: Float, mean that the constraint shoud realise 
            u_0: example input for the model, only its dimensions are taken

        Returns: 
            1x1xHxW tensor with the mean mu 
        """
        return mu * torch.ones(1, 1, u_0.size(2), u_0.size(3), dtype=u_0.dtype, device=u_0.device)



class MeanConstraintPseudoinverse(ConstraintPseudoinverse):

    def __init__(self, u_0, t_0=0.) -> None:
        pass

    def jacobian(self, u, t=0.):
        """Implements the mean constraint, the Jacobian is just ones 
        
        """
        N_batch = u.size(0)
        H = u.size(2)
        W = u.size(3)

        return torch.ones(N_batch, 1, H*W, device=u.device, dtype=u.dtype) 


class BoundaryConstraint():
    def __init__(self, u, t=0.) -> None:
        self.boundary = self.get_border_points(u)
        self.boundary_flat = self.boundary.flatten()
        self.N_boundary = self.boundary.sum()

    def get_border_points(self, u):
        boundary_mask = torch.ones(u.size(2),u.size(3),dtype=bool, device=u.device)
        boundary_mask[1:-1,1:-1] = False 
        return boundary_mask

class BoundaryConstraintFunction(ConstraintFunction, BoundaryConstraint):

    def __init__(self, u_0, t_0=0.) -> None:
        super().__init__(u_0, t_0)

        self.val_0 = self.function(u_0, t_0)

    def function(self, u, t=0.):
        """Implements the boundary constraint
        
        Args: 
            u: Tensor Nx1xHxW input argument
            t: Float time, ignored 

        Returns:
            Values at the boundary as N x m_boundary  
        """
        return u[:,0,self.boundary]
    
    @staticmethod 
    def initialcondition(boundary_val, u):
        """Helper function to prepare an 'initial condition' for the BoundaryConstraintFunction 
        
        Args: 
            boundary_val: Float, value at the boundary that the constraint should realise 
            u: example input for the model, only its dimensions are taken

        Returns: 
            1x1xHxW tensor with the boundary at value boundary_val
        """
        boundary = BoundaryConstraint(None, u, 0)
        ic = torch.zeros(1,1,u.size(2),u.size(3),dtype=u.dtype, device=u.device)
        ic[:,:,boundary.boundary] = boundary_val
        return ic 

class BoundaryConstraintPseudoinverse(ConstraintPseudoinverse, BoundaryConstraint):

    def __init__(self, u_0, t_0=0.) -> None:
        super().__init__(u_0, t_0)

        self.jac_matrix = self.construct_jacobian_matrix(u_0)

    def construct_jacobian_matrix(self, u_0):
        """Constructs the Jacobian matrix for the boundary constraint. 

        Args:
            u_0: Tensor Nx1xHxW initial condition for the constraint 

        Returns: 
            Jacobian tensor 1 x m x (H*W) - m is the number of boundary points 
        """
        
        H = u_0.size(2)
        W = u_0.size(3)

        jac = torch.zeros(1, self.N_boundary, H*W, dtype=u_0.dtype, device=u_0.device)

        boundary_counter = 0
        for i_b in range(H*W):
            if self.boundary_flat[i_b]:
                jac[0, boundary_counter, i_b] = 1 
                boundary_counter += 1 

        return jac 
    
    def jacobian(self, u, t):
        return self.jac_matrix.repeat(u.size(0), 1, 1)

class DownscalingConstraintFunction(ConstraintFunction): 

    def __init__(self, u_0, t_0=0., pooling_kernel=(2,2)) -> None:
        """Initialized a downscaling constraint, so that the downscaled pixels sum to the same value of the original image. 
        
        Args. 
            u_0: ATTENTION!, here the initial condition is in the coarse resolution 
            t_0: Float, Not used 
            pooling kernel: tuple, the kernel size of the pooling that is used for getting from the fine resolution to the coarse resolution 
        """
        self.pooling = torch.nn.AvgPool2d(pooling_kernel)
        self.val_0 = torch.reshape(u_0,(u_0.size(0),-1))

    def remake(self, u, t=0.): 
        """Sets a new constraint reference for the downscaling task 

        Args 
            u: tensor 1x1xHxW 
        """
        self.val_0 = torch.reshape(u, (u.size(0),-1))

    def function(self, u, t):
        """Implements the 4x downscaling constraint
        
        Args: 
            u: Tensor Nx1xHxW input argument
            t: Float time, ignored 

        Returns:
            Avg. Pooled input 
        """
        return torch.reshape(self.pooling(u),(u.size(0),-1)) # missing some flattinging 

class DownscalingConstraintPseudoinverse(ConstraintPseudoinverse):

    def __init__(self, u_0, t_0, pooling_kernel=(2,2)) -> None:

        self.pooling_kernel = pooling_kernel
        self.jac_matrix = self.construct_jacobian_matrix(u_0)

    def construct_jacobian_matrix(self, u_0):

        N_batch = 1 # dummy value, correct one is used in jacobian function
        H_small = u_0.size(2)
        W_small = u_0.size(3)
        H = H_small * self.pooling_kernel[0]
        W = W_small * self.pooling_kernel[1]
        N_small = H_small*W_small

        J = torch.zeros(N_batch, H_small, W_small, H, W, device=u_0.device, dtype=u_0.dtype)
        
        for i in range(H_small):
            for j in range(W_small):
                J[:,i,j,2*i:2*(i+1),2*j:2*(j+1)] = 1

        J = torch.reshape(J, (N_batch, H_small*W_small, H, W))

        return torch.reshape(J, (N_batch, N_small, H*W))

    def jacobian(self, u, t):
        return self.jac_matrix.repeat(u.size(0), 1, 1)
