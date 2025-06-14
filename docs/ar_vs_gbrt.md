# AR Models vs Gradient Boosting

## AR(1)

An AR(1) is a function over the last measured step: $Y_t = f(Y_{t-1})$

### $AR(1)$ model implementation:

$Y_t = c + \phi Y_{t-1} + \varepsilon_{t}$

Where:
* $Y_t$: Value at time t
* $c$: Constant term (intercept)
* $\phi$: Autoregressive coefficient
* $Y_{t-1}$: Value at previous time step
* $\varepsilon_{t}$: White noise error term

### Gradient boosting model alternative:

An gradient boosted regression for the same data can be expressed as:

$Y_t = F_0(Y_{t-1}) + \sum_{m=1}^{M} \alpha_m h_m(Y_{t-1}; \mathbf{\theta}_m)$

Where:
* $Y_t$: Value at time t
* $F_M(Y_{t-1})$: The final model after M boosting iterations
* $F_0(Y_{t-1})$: The initial model (before any boosting)
* $\alpha_{m}$: The weight (or learning rate) associated with the m-th weak learner.
* $h_m(Y_{t-1}; \mathbf{\theta}_m)$: The m-th weak learner, which is a function of the input x and is parameterized by $θ_{m}$