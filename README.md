
# PyHKBs

Multi-HKBs agents in Python

## Roadmap

- Initial experiment with toy model [2022-07-05, [helloworld.py](helloworld.py)]

- Implement HKB equation and illustrate properties [2022-07-06, [basicHKB.py](basicHKB.py)]

- Implement HKB composed of two oscillators [2022-07-07, [two_oscillators.py](two_oscillators.py)]

- Simulate a network of oscillators with a HKB coupling matrix ($a_{i,j}, b_{i,j}$) [2022-07-11, [network_of_oscillators.py](network_of_oscillators.py)]

- Create an agent class with input and output [2022-07-13, [agent.py](agent.py)]

- Build a simple environment (matplotlib) [2022-07-14, [single_agent_simulation.py](agent.py)]

- Use artificial evolution to find optimal HKB coupling matrix

- Build a more complex environment by interfacing with Unity 3D

## Agent design
![alt text](https://github.com/ppsp-team/PyHKBs/blob/main/agentSchema.png?raw=true)
The agent has four oscillators, corresponding to the sensory system (1 and 2) and motor system (3 and 4). The grey boxes represent the two eyes (or sensors). The left eye feeds the change in stimulus intensity to oscillator 1; the right eye to oscillator 2. The agent's orientation in space changes according to the phase difference between the two motor oscillators. The agent travels with a uniform speed. 

## References

Aguilera, M., Bedia, M. G., Santos, B. A., & Barandiaran, X. E. (2013). The situated HKB model: how sensorimotor spatial coupling can alter oscillatory brain dynamics. Frontiers in computational neuroscience, 7, 117. [doi:10.3389/fncom.2013.00117](https://www.frontiersin.org/articles/10.3389/fncom.2013.00117/full)

Frank, T. D., Daffertshofer, A., Peper, C. E., Beek, P. J., & Haken, H. (2000). Towards a comprehensive theory of brain activity:: Coupled oscillator systems under external forces. Physica D: Nonlinear Phenomena, 144(1-2), 62-86. [doi:10.1016/S0167-2789(00)00071-3](https://www.sciencedirect.com/science/article/pii/S0167278900000713?via%3Dihub)

Haken, H., Kelso, J. S., & Bunz, H. (1985). A theoretical model of phase transitions in human hand movements. Biological cybernetics, 51(5), 347-356. [doi:10.1007/BF00336922](https://link.springer.com/article/10.1007/BF00336922)

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., ... & Hochreiter, S. (2020). Hopfield networks is all you need. arXiv preprint arXiv:2008.02217. [doi:10.48550/arXiv.2008.02217](https://arxiv.org/abs/2008.02217)

Zhang, M., Beetle, C., Kelso, J. S., & Tognoli, E. (2019). Connecting empirical phenomena and theoretical models of biological coordination across scales. Journal of the Royal Society Interface, 16(157), 20190360. [doi:10.1098/rsif.2019.0360](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2019.0360)
