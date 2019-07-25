Research
########

:order: 2
:date: 2018-08-15 12:00
:icon: icon-lightbulb
:summary: Things that I've been thinking about
:lang: en
:slug: research

Research Interests
~~~~~~~~~~~~~~~~~~

My research interests and experience lie at the intersection of deep learning, computational modeling,
and complex systems. Specifically, I'm interested in using deep learning methods to better
understand and model complex and dynamic data.

|

Deep Learning + Computational Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Learning nonlinear reduced order models
---------------------------------------
.. container:: float-left

    .. image:: /images/research/learning_lid.gif
        :width: 250px


Many fields, particularly in the physical sciences, are in the fortunate position of
having models based on first-principles that describe the evolution of certain
systems with near-perfect accuracy. Notable examples include the Navier-Stokes
equations in fluid mechanics or Schrödinger's equation in quantum mechanics.
Although in principle it is possible to numerically solve these equations
through direct numerical simulations, this often yields systems of equations
with millions or billions of degrees of freedom. Even with recent advances in
computational power and memory capacity solving these high-fidelity models is
still computationally intractable for *multi-query* and *time-critical*
applications such as design optimization, uncertainty quantification, and model
predictive control.

Although there are a wide variety of principled strategies for constructing
reduced order models from data, most are intrusive as they require access to the
system operators. Further, some systems may require special treatment of
nonlinearities to ensure computational efficiency or additional modeling to
preserve stability. The recent rise in deep learning and big data have driven a
shift in the way complex spatiotemporal systems are modeled. In a recent work titled
`Deep convolutional recurrent autoencoders for learning low-dimensional feature dynamics of fluid systems <https://arxiv.org/abs/1808.01346>`_
we develop a deep learning based, completely data-driven approach to model
reduction. At the core of this approach is an extension of projection-based
model reduction in which a low-dimensional representation of the
high-dimensional data is *learned* in the form of coordinates on an expressive
nonlinear manifold. The dynamics of this representation on the underlying
manifold are also learned using a representative collection of solution
snapshots.




Deep dilation models for multi-scale dynamics
---------------------------------------------
.. container:: float-right

    .. image:: /images/research/wavenet.png
        :width: 375px

Recurrent neural networks (RNNs) have long been used to model sequential or
time-dependent data. However, many real-world physical systems, e.g., turbulent
flow in fluids, exhibit multi-scale dynamics. Thus, while RNNs can accurately
model the dynamics of a system at one time-scale, they fail to capture the
dynamics occurring over a multitude of scales. This problem is not unique to
physical systems: RNNs also fail to capture the varying time-scales evident in
speech data. Recent work involving models with dilated RNNs and convolutional
networks, e.g., `WaveNet <https://arxiv.org/abs/1711.10433>`_ (see figure), have
shown great performance in modeling multi-scale speech data. Currently, I am
developing deep dilation models inspired by WaveNet to help improve deep
learning based modeling of multi-scale physical systems. This work attempts to
explore the how dilated RNNs help capture dynamics at different scales, and thereby
significantly increasing the applicability of deep learning based modeling approaches
to real-world dynamical systems.

|

High-Performance Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Quinoa: Adapative Computational Fluid Dynamics
----------------------------------------------

.. container:: float-right

    .. image:: /images/research/chare_migration.png
        :width: 250px

As we enter the exascale era of high-performance computing performance of
computational physics codes will increasingly depend on their ability to
asynchronously adapt to varying computational loads induced by multi-scale and
multi-physics simulations as well as varying hardware performance. During my
time at Los Alamos National Laboratory I contributed to the development of
`Quinoa <https://quinoacomputing.github.io>`_, an open-source computational
science code specifically addressing the challenges that will be faced with
heterogeneous exascale machines. Quinoa is built on top of the Charm++ runtime
system which allows for asynchronous parallel execution enabling the
overlapping of computation, communication, and I/O. In future work, I would like
to incorporate deep learning based methods into Quinoa for asynchronous parallel
modeling based solely on multi-scale and multi-physics data.
