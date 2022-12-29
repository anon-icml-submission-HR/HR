import cvxpy as cp
import torch
import numpy as np
from torch.autograd import Variable
import torchattacks
import warnings
import mosek
import os
os.environ['MOSEKLM_LICENSE_FILE'] = "mosek.lic"


class HR_Neural_Networks:

    def __init__(self, NN_model,
                 learning_approach,
                 train_batch_size,
                 loss_fn,
                 normalisation_used,
                 α_choice,
                 r_choice,
                 ϵ_choice,
                 adversarial_steps=10,
                 adversarial_step_size=0.2,
                 noise_set = "l-2"
                 ):

        # End model
        self.NN_model = NN_model
        self.train_batch_size = train_batch_size
        self.adversarial_steps = adversarial_steps
        self.adversarial_step_size = adversarial_step_size
        self.numerical_eps = 0.000001
        self.noise_set = noise_set
        self.learning_approach = learning_approach
        
        self.primary_solver = cp.ECOS # Fastest solver for conic problems
        self.secondary_solver = cp.MOSEK # Commercial solver (faster when r = 0)
        self.third_solver = cp.SCS # Most reliable for conic problems, but also slowest

        if loss_fn == None:
            raise warnings.warn(
                "Loss is defaulted to cross entropy loss. Consider changing if not doing classification.")
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        else:
            self.loss_fn = loss_fn
        
        # Handling choice of α
        if α_choice == 0:
            self.α_choice = self.numerical_eps
        else:
            self.α_choice = α_choice
        
        # Handling choice of r
        if r_choice == 0:
            self.r_choice = self.numerical_eps
            self.primary_solver = cp.MOSEK
        else:
            self.r_choice = r_choice
            
        # Handling choice of epsilon. We wont set equal to numerical eps, since running PGD is very slow
        self.ϵ_choice = ϵ_choice
        
        # Initialising either HR or HD to be used in DPP. DPP is an approach where the decision variables,
        # constraints and problem are set up just once. Only parameters (here loss and worst-case) 
        # are reinitialised at each step, which is much faster than reinstating the entire problem. 
        if self.learning_approach == "HR":

            self._initialise_HR_problem()

        elif self.learning_approach == "HD":
            
            self._initialise_HD_problem()

        self._initialise_adversarial_setup()

        if normalisation_used == None:
            raise warnings.warn(
                "Warning - data normalisation is defaulted to CIFAR-10. If you are using another dataset pls change.")
            self.adversarial_attack_train.set_normalization_used(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        else:
            self.adversarial_attack_train.set_normalization_used(
                mean=normalisation_used[0], std=normalisation_used[1])

    def _initialise_HR_problem(self):

        # The primal - inner maximisation problem.
        N = self.train_batch_size

        Pemp = 1/N * np.ones(N)  # Change for a diffrent Pemp

        # Parameter controlling robustness to misspecification
        α = cp.Constant(self.α_choice)
        # Parameter controlling robustness to statistical error
        r = cp.Constant(self.r_choice)

        # Primal variables and constraints, indep of problem
        self.p = cp.Variable(shape=N+1, nonneg=True)
        q = cp.Variable(shape=N+1, nonneg=True)
        s = cp.Variable(shape=N, nonneg=True)

        self.nn_loss = cp.Parameter(shape=N)
        self.nn_loss.value = [1/N]*N  # Initialising

        self.worst = cp.Parameter()
        self.worst.value = 0.01  # Initialising

        # Objective function
        objective = cp.Maximize(
            cp.sum(cp.multiply(self.p[0:N], self.nn_loss)) + self.p[N] * self.worst)

        # Simplex constraints
        simplex_constraints = [cp.sum(self.p) == 1, cp.sum(q) == 1]

        # KL constr -----
        t = cp.Variable(name="t", shape=N)

        # Exponential cone constraints
        exc_constraints = []

        exc_constraints.append(
            cp.constraints.exponential.ExpCone(-1*t, Pemp, q[:-1]))

        # ------------------------
        extra_constraints = [cp.sum(t) <= r,
                             cp.sum(s) <= α,
                             cp.sum(s) + q[N] == self.p[N],
                             self.p[0:N] + s == q[0:N]]
        # ------------------------

        # Combining constraints to a single list
        complete_constraints = simplex_constraints + exc_constraints + extra_constraints

        # Problem definition
        self.model = cp.Problem(
            objective=objective,
            constraints=complete_constraints)

    def _initialise_HD_problem(self):

        # The primal - inner maximisation problem.
        N = self.train_batch_size

        Pemp = 1/N * np.ones(N)  # Change for a diffrent Pemp

        # Parameter controlling robustness to misspecification
        α = cp.Constant(self.α_choice)
        # Parameter controlling robustness to statistical error
        r = cp.Constant(self.r_choice)

        # Primal variables and constraints, indep of problem
        self.p = cp.Variable(shape=N+1, nonneg=True)
        q = cp.Variable(shape=N+1, nonneg=True)
        s = cp.Variable(shape=N, nonneg=True)

        self.nn_loss = cp.Parameter(shape=N)
        self.nn_loss.value = [1/N]*N  # Initialising

        self.worst = cp.Parameter()
        self.worst.value = 0.01  # Initialising

        # Objective function
        objective = cp.Maximize(
            cp.sum(cp.multiply(self.p[0:N], self.nn_loss)) + self.p[N] * self.worst)

        # Simplex constraints
        simplex_constraints = [cp.sum(self.p) == 1, cp.sum(q) == 1]

        # KL constr -----
        t = cp.Variable(name="t", shape=N+1)

        # Exponential cone constraints
        exc_constraints = []

        exc_constraints.append(
            cp.constraints.exponential.ExpCone(-1*t, q, self.p))

        # ------------------------
        extra_constraints = [cp.sum(t) <= r,
                             cp.sum(s) <= α,
                             q[0:N] + s == Pemp]
        # ------------------------

        # Combining constraints to a single list
        complete_constraints = simplex_constraints + exc_constraints + extra_constraints

        # Problem definition
        self.model = cp.Problem(
            objective=objective,
            constraints=complete_constraints)

    def _initialise_adversarial_setup(self):
        
        if self.noise_set == "l-2":

            self.adversarial_attack_train = torchattacks.PGDL2(self.NN_model,
                                                               eps=self.ϵ_choice,
                                                               alpha=self.adversarial_step_size,
                                                               steps=self.adversarial_steps,
                                                               random_start=True,
                                                               eps_for_division=1e-10)
            
        elif self.noise_set == "l-inf":
            
            self.adversarial_attack_train = torchattacks.attacks.pgd.PGD(self.NN_model,
                                             eps=self.ϵ_choice,
                                             alpha=self.adversarial_step_size,
                                             steps=self.adversarial_steps,
                                             random_start=True)


    def HR_criterion(self, inputs, targets, device='cuda'):
        
        '''Solving the primal problem.
           Returning the weighted loss as a tensor Pytorch can autodiff'''

        if self.ϵ_choice > 0:
            adv = self.adversarial_attack_train(inputs, targets)
        else:
            adv = inputs

        outputs = self.NN_model(adv)
        inf_loss = self.loss_fn(outputs, targets)
        
        # May in the end be different from the training batch size,
        batch_size = len(inf_loss)
        # For instance for the last batch
        
        if batch_size != self.train_batch_size: # If the batches passed are not the same length as the pre-specified
                                                # train_batch_size, then we need to renitialise the DPP.
                                                # DPP assumes certain parameters of the problem (here, N) remain fixed.
                    
            warnings.warn(
                "Warning - changing the batch_size from the pre-specified train_batch_size can cause the algorithm to be slower.")
            
            self.train_batch_size = batch_size
            
            if self.learning_approach == "HR":

                self._initialise_HR_problem()

            elif self.learning_approach == "HD":

                self._initialise_HD_problem()

        if self.r_choice > self.numerical_eps or self.α_choice > self.numerical_eps:

            self.nn_loss.value = np.array(inf_loss.cpu().detach().numpy()) # DPP step
            self.worst.value = np.max(self.nn_loss.value) # DPP step

            try:
                self.model.solve(solver=self.primary_solver) 
                # ECOS is normally faster than MOSEK for conic problems (it is built for this purpose),
                # but generally also more unstable. 
                # We will revert to MOSEK incase of solving issues.
                # This should happen very infrequently (<1/1000 calls or so, depending on α, r)
                
            except:
                self.nn_loss.value += self.numerical_eps # Small amt of noise incase its a numerical issue
                self.worst.value = np.max(self.nn_loss.value) # Must also re-instate worst-case for DPP
                self.model.solve(solver=self.secondary_solver)
                # MOSEK is the second fastest,
                # But also occasionally fails when α and r are too large.
            finally:
                self.model.solve(solver=self.third_solver)
                # SCS is the slowest,
                # but also the most reliable for conic problems.

            weights = Variable(torch.from_numpy(self.p.value),
                               requires_grad=True).to(device).float() # Converting primal weights to tensors

            return torch.dot(weights[0:batch_size], inf_loss) + torch.max(inf_loss)*weights[batch_size]

        else: # If we use only epsilon (could be zero or not)
            return (1/self.train_batch_size)*torch.sum(inf_loss)
