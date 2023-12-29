# Awesome-NN-ConstrainedOptimization
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Awesome Model Merging](https://img.shields.io/badge/Awesome-Model--Merging-blue)](https://github.com/topics/awesome)

A curated list of **NN ConstrainedOptimization** papers: Using neural networks to solve constrained optimization problems.

**Contributions are welcome!**

*Acknowledgments:
This wonderful template is from https://github.com/VainF/Awesome-Anything by [Gongfan Fang](https://github.com/VainF).*


## General methods/framework
| Title & Authors | Intro | Useful Links |
|:----|  :----: | :---:|
|[**DC3: A learning method for optimization with hard constraints**](https://arxiv.org/abs/2104.12225) <br> *P. Donti, D. Rolnick, J. Z. Kolter* <br> >   ICLR'21<br><br> | This method enforces feasibility via a differentiable procedure, which implicitly completes partial solutions to satisfy equality constraints and unrolls gradient-based corrections to satisfy inequality constraints. We demonstrate the effectiveness of DC3 in both synthetic optimization tasks and the real-world setting of AC optimal power flow, where hard constraints encode the physics of the electrical grid. In both cases, DC3 achieves near-optimal objective values while preserving feasibility. | [[Github](https://github.com/locuslab/DC3)] <br> [[PDF](https://arxiv.org/pdf/2104.12225.pdf)] |
|[**Ensuring DNN Solution Feasibility for Optimization Problems with Convex Constraints and Its Application to DC Optimal Power Flow Problems**](https://arxiv.org/abs/2112.08091) <br> *Tianyu Zhao, Xiang Pan, Minghua Chen, Steven H. Low* <br> >   ICLR'23 Oral<br><br> | This method enforces feasibility via a differentiable procedure, which implicitly completes partial solutions to satisfy equality constraints and unrolls gradient-based corrections to satisfy inequality constraints. We demonstrate the effectiveness of DC3 in both synthetic optimization tasks and the real-world setting of AC optimal power flow, where hard constraints encode the physics of the electrical grid. In both cases, DC3 achieves near-optimal objective values while preserving feasibility. | [[Github](https://github.com/locuslab/DC3)] <br> [[PDF](https://arxiv.org/pdf/2112.08091.pdf)] |




## Optimal power flow
| Title & Authors | Intro | Useful Links |
|:----|  :----: | :---:|
|[**DeepOPF: Deep Neural Network for DC Optimal Power Flow**](https://arxiv.org/abs/1905.04479) <br> *Xiang Pan, Tianyu Zhao, and Minghua Chen* <br> >  2019, IEEE International Conference on Smart Grid Communications <br><br> | DeepOPF leverages a DNN model to depict the high-dimensional load-to-solution mapping and can directly solve the OPF problem upon given load, excelling in fast computation process and desirable scalability. | [[Github](https://github.com/Mzhou-cityu/DeepOPF-Codes)] <br> [[PDF](https://arxiv.org/pdf/1905.04479.pdf)] |
|[**Optimal Power Flow Using Graph Neural Networks**](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053140.pdf?casa_token=TaGU0aOOMNkAAAAA:RmQzI9WV_y-FePGJstSWIzqBNCS0t2kEnLHiw-Wri56HKdTBkvHd4EQa0HWdwvGWAoziV3cGEgI) <br> *D Owerko, F Gama, A Ribeiro* <br> >   ICASSP'20 <br><br> | This method uses imitation learning and graph neural networks to find a local and scalable solution to the OPF problem.  |  <br> [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053140)] |
|[**Learning Optimal Power Flow: Worst-Case Guarantees for Neural Networks**](https://arxiv.org/abs/2006.11029) <br> *Andreas Venzke, Guannan Qu* <br> >   Arxiv <br><br> | This paper introduces for the first time a framework to obtain provable worst-case guarantees for neural network performance, using learning for optimal power flow (OPF) problems as a guiding example.  |  <br> [[PDF](https://arxiv.org/pdf/2006.11029.pdf)] |



##  Combinatorial optimization problem
| Title & Authors | Intro | Useful Links |
|:----|  :----: | :---:|
|[**Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching**](https://arxiv.org/abs/2012.08950) <br> *Chang Liu, Zetian Jiang, Runzhong Wang, Junchi Yan, Lingxiao Huang, Pinyan Lu* <br> > SJTU <br> >  ICLR'23 <br><br> | Towards practical and robust graph matching learning, in the absence of labels and in the presence of outliers (in both input graphs), this paper proposes a reinforcement learning method for graph matching namely RGM, especially for its most general QAP formulation | [[Github](https://github.com/Thinklab-SJTU/RGM)] <br> [[PDF](https://arxiv.org/pdf/2012.08950.pdf)] |
|[**Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization**](https://arxiv.org/abs/2210.03475) <br> *Nathan Grinsztajn, Daniel Furelos-Blanco, Shikha Surana, Clément Bonnet, Thomas D. Barrett*  <br> >  NIPS'23 <br><br> | This paper introduces Poppy, a simple training procedure for populations. Instead of relying on a predefined or hand-crafted notion of diversity, Poppy induces an unsupervised specialization targeted solely at maximizing the performance of the population. We show that Poppy produces a set of complementary policies, and obtains state-of-the-art RL results on four popular NP-hard problems: traveling salesman, capacitated vehicle routing, 0-1 knapsack, and job-shop scheduling. |  <br> [[PDF](https://arxiv.org/pdf/2210.03475.pdf)] |
|[**A GNN-Guided Predict-and-Search Framework for Mixed-Integer Linear Programming**](https://arxiv.org/abs/2302.05636) <br> *Qingyu Han, Linxin Yang, Qian Chen, Xiang Zhou, Dong Zhang, Akang Wang, Ruoyu Sun, Xiaodong Luo* <br> >  Shenzhen Research Institute of Big Data <br> >  ICLR'23 <br><br> | This paper proposes a novel predict-and-search framework for efficiently identifying high-quality feasible solutions. Specifically, they first utilize graph neural networks to predict the marginal probability of each variable, and then search for the best feasible solution within a properly defined ball around the predicted solution | [[Github](https://github.com/sribdcn/Predict-and-Search_MILP_method)] <br> [[PDF](https://arxiv.org/pdf/2302.05636.pdf)] |
|[**CktGNN: Circuit Graph Neural Network for Electronic Design Automation**](https://arxiv.org/abs/2308.16406) <br> *Zehao Dong, Weidong Cao, Muhan Zhang, Dacheng Tao, Yixin Chen, Xuan Zhang* <br> >  WUSTL <br> >  ICLR'23 <br><br> | CktGNN is a two-level GNN model with a pre-designed subgraph basis for the analog circuit (DAG) encoding. CktGNN simultaneously optimizes circuit topology and device features, achieving state-of-art performance in analog circuit optimization. | [[Github](https://github.com/zehao-dong/CktGNN)] <br> [[PDF](https://arxiv.org/pdf/2308.16406.pdf)] |
|[**Two-Stage Predict+Optimize for Mixed Integer Linear Programs with Unknown Parameters in Constraints**](https://arxiv.org/abs/2311.08022) <br> *Xinyi Hu, Jasper C.H. Lee, Jimmy H.M. Lee* <br> >  CUHK <br> >  NIPS'23 <br><br> | Predict+Optimize is a recent framework for end-to-end training supervised learning models for such predictions, incorporating information about the optimization problem in the training process in order to yield better predictions in terms of the quality of the predicted solution under the true parameters. They also give a training algorithm usable for all mixed integer linear programs, vastly generalizing the applicability of the framework | [[Github](https://github.com/Elizabethxyhu/NeurIPS_Two_Stage_Predict-Optimize)] <br> [[PDF](https://arxiv.org/pdf/2311.08022.pdf)] |
|[**Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets**](https://arxiv.org/abs/2305.17010) <br> *Dinghuai Zhang, Hanjun Dai, Nikolay Malkin, Aaron Courville, Yoshua Bengio, Ling Pan* <br> >  Mila <br> >  NIPS'23 <br><br> | This paper designs Markov decision processes (MDPs) for different combinatorial problems and proposes to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment. | [[Github](https://github.com/zdhNarsil/GFlowNet-CombOpt)] <br> [[PDF](https://arxiv.org/pdf/2305.17010.pdf)] |

## Physics-Inspired Optimization
| Title & Authors | Intro | Useful Links |
|:----|  :----: | :---:|
|[**Combinatorial Optimization with Physics-Inspired Graph Neural Networks**](https://arxiv.org/abs/2107.01188) <br> *Martin J. A. Schuetz, J. Kyle Brubaker, Helmut G. Katzgraber <br> > Amazon <br> >  Nature Machine Intelligence '22 <br><br> | This paper uses Physics-inspired GNN to solve quadratic unconstrained binary optimization problems, such as maximum cut, minimum vertex cover, maximum independent set, as well as Ising spin glasses and higher-order generalizations thereof in the form of polynomial unconstrained binary optimization problems.  | [[Github](https://github.com/amazon-research/co-with-gnns-example)] <br> [[PDF](https://arxiv.org/pdf/2107.01188.pdf)] |
## Using penalty terms to handle constrained optimization problems 


... (TBD)
