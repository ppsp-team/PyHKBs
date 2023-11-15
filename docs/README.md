
# PyHKBs

This repository contains the code for simulating embodied neural agents controlled by the Haken-Kelso-Bunz (HKB) equations.
The code uses Pytorch to enable parallelized solving of differential equations.

## Script functionalities
- [basicHKB.py](basicHKB.py) Illustration of how to solve the HKB equations with PyTorch
- [two_oscillators.py](two_oscillators.py) Animation of two interacting oscillators with HKB equations
- [agent_RL.py](agent_RL.py) Contains classes of agents 
- [environment.py](environment.py) Contains classes of environments in which the agents move
- [simulations.py](simulations.py) Contains functions to perform agent runs with agent and environment classes
- [agent_evaluation.py](agent_evaluation.py) Executes single run simulations of single agents and plots trajectories
- [social_agent_evaluation.py](social_agent_evaluation.py) Executes single run simulations of multiple agents and plots trajectories
- [grid_search.py](grid_search.py) Executes and saves single_agent runs for many different parameter values
- [grid_search_social.py](grid_search_social.py) Executes and saves single_agent runs for many different parameter values
- [grid_search_evaluation.py](grid_search.py) Evaluates the saved runs of the single agent grid search
- [grid_search_social_evaluation.py](grid_search_social_evaluation.py) Evaluates the saved runs of the multi-agent grid search
- [visualizations.py](visualizations.py) Helper functions for plotting simulated runs
- [ternary_plot.py](ternary_plot.py) Example code for generating ternary plots
- [utils.py](utils.py) Helper functions
- [training.py](utils.py) Helper functions

## Agent design
![alt text](https://github.com/ppsp-team/PyHKBs/docs/agentSchema.png?raw=true)
Our agent design (named 'Guido' for 'guided oscillator' in [agent_RL.py](agent_RL.py)) consists of four oscillators, corresponding to the sensory system (1 and 2) and motor system (3 and 4). The grey boxes represent the two eyes (or sensors). The left eye feeds the change in stimulus intensity to oscillator 1; the right eye to oscillator 2. The agent's orientation in space changes according to the phase difference between the two motor oscillators. The agent travels at a uniform speed. Both the simulations with individual agents and the multi-agent simulations in the main paper are run with four-oscillator agents. The [agent_RL.py](agent_RL.py) also contains a 'SocialGuido' class with an additional 5th oscillator that represents a 'socially sensitive' oscillator that is directly sensitive to the phases of other agents. This class has not been used in the main paper.


## Note on training possibilities
The environment classes in [environment.py](environment.py) and the agent classes in [agent_RL.py](agent_RL.py) are made with a similar structure as standard reinforcement learning approaches. 
The different layers of the agent classes in [agent_RL.py](agent_RL.py) are all differentiable using Autograd to ensure that they can possibly be trained using policy gradient methods in Pytorch. 

## Cite as
Coucke, N., Heinrich, M. K., Cleeremans, A., Dorigo, M. & Dumas, G. (2023). Collective decision making with embodied neural agents (in prep).

## References

Aguilera, M., Bedia, M. G., Santos, B. A., & Barandiaran, X. E. (2013). The situated HKB model: how sensorimotor spatial coupling can alter oscillatory brain dynamics. Frontiers in computational neuroscience, 7, 117. [doi:10.3389/fncom.2013.00117](https://www.frontiersin.org/articles/10.3389/fncom.2013.00117/full)

Frank, T. D., Daffertshofer, A., Peper, C. E., Beek, P. J., & Haken, H. (2000). Towards a comprehensive theory of brain activity:: Coupled oscillator systems under external forces. Physica D: Nonlinear Phenomena, 144(1-2), 62-86. [doi:10.1016/S0167-2789(00)00071-3](https://www.sciencedirect.com/science/article/pii/S0167278900000713?via%3Dihub)

Haken, H., Kelso, J. S., & Bunz, H. (1985). A theoretical model of phase transitions in human hand movements. Biological cybernetics, 51(5), 347-356. [doi:10.1007/BF00336922](https://link.springer.com/article/10.1007/BF00336922)

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., ... & Hochreiter, S. (2020). Hopfield networks is all you need. arXiv preprint arXiv:2008.02217. [doi:10.48550/arXiv.2008.02217](https://arxiv.org/abs/2008.02217)

Zhang, M., Beetle, C., Kelso, J. S., & Tognoli, E. (2019). Connecting empirical phenomena and theoretical models of biological coordination across scales. Journal of the Royal Society Interface, 16(157), 20190360. [doi:10.1098/rsif.2019.0360](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2019.0360)



- Use artificial evolution or RL to find optimal HKB coupling matrix

- Build a more complex environment by interfacing with Unity 3D
