
import numpy as np
import scipy
from scipy import optimize
import random
import pandas



def cmrt(nu, z, a, Ter=0, s = 0.1, GRAD=False):
    """
    Compute exit (decision) time mean and variance _conditional_ on exit 
    throught the bottom boundary of a diffusion process between absorbing
    boundaries.

    Description:

         Given a boundary separation, a starting point, and a drift rate,
         this function computes the mean exit time/exit time variance of a
         one dimensional diffusion process under constant drift on an
         interval with absorbing boundaries, conditioned on the
         exit being through the bottom boundary (i.e., error responses). 
         Used as a model of information accumulation, it is gives the
         mean decision time/decision time variance of responses in a
         speeded two-alternative forced choice (2AFC) response time task,
         conditional on what alternative was decided upon.
         
         Note: For top boundary exits use ‘a - z’ as starting point.

    Usage:

         cmrt(nu, z, a, s = 0.1, Ter=0)   # bottom boundary exits
         cvrt(nu, z, a, s = 0.1, Ter=0)   # bottom boundary exits

         cmrt(nu, a-z, a, s = 0.1, Ter=0) # top boundary exits
         cvrt(nu, a-z, a, s = 0.1, Ter=0) # top boundary exits

    Arguments:

          nu: Float. Drift rate.

           z: Float. Starting point.

           a: Float. Boundary separation

           s: Scaling parameter (Ratcliff's convention is ‘s = 0.1’, the
              default)
              
         Ter: Lag (models non-decision time of a response time)
         
        GRAD: Logical. Should the gradient be returned?

    Details:

         This model of information accumulation and decision is a
         simplified version of Ratcliff's diffusion model (1978). It can be
         used, e.g., to compute the mean response times of the correct
         responses in a lexical decision time, given the drift rate, the
         bias (start point), and criterion (boundary separation).

    Value:

         ‘cmrt’ returns the mean exit/decision time(s) ‘cvrt’    
         returns the exit/decision time variance(s). The return value has
         the attribute "gradient" attached: the gradient with respect to
         each of the parameters.

    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108.

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘mrt’, ‘vrt’, ‘batch’

    Examples:

             mrt( 0.1, 0.08, 0.12, Ter=0.25)
             vrt( 0.1, 0.08, 0.12)
    
    
    """
    import numpy as np
    from numpy import exp
    nu = np.array(nu)
    z = np.array(z)
    a = np.array(a)
        
    expr2 = s**2
    expr3 = 4 * nu/expr2
    expr5 = exp(expr3 * a)
    expr6 = 2 * nu
    expr7 = z + a
    expr10 = exp(expr6 * expr7/expr2)
    expr12 = expr6/expr2
    expr14 = exp(expr12 * a)
    expr18 = exp(expr6 * z/expr2)
    expr19 = expr5 + expr10 - expr14 - expr18
    expr23 = 2 * expr14 - 2 * expr10
    expr25 = expr19 * z + expr23 * a
    expr26 = expr25/nu
    expr27 = expr14 - expr18
    expr28 = expr26/expr27
    expr30 = -1 + expr14
    expr37 = expr10 * (2 * expr7/expr2)
    expr41 = expr14 * (2/expr2 * a)
    expr45 = expr18 * (2 * z/expr2)
    expr60 = expr27**2
    expr65 = expr30**2
    expr68 = expr10 * expr12
    expr69 = expr18 * expr12
    expr73 = 2 * expr68
    expr84 = expr14 * expr12
    value = expr28/expr30 + Ter
    
    m = np.size(value)
    grad = np.array([0 for i in range(m)], dtype=[('nu', 'float'),('z', 'float'),('a', 'float'),('Ter', 'float')])
    
    grad["nu"] = ((((expr5 * (4/expr2 * a) + expr37 - 
        expr41 - expr45) * z + (2 * expr41 - 2 * expr37) * 
        a)/nu - expr25/nu**2)/expr27 - expr26 * (expr41 - 
        expr45)/expr60)/expr30 - expr28 * expr41/expr65
    grad["z"] = (((expr68 - expr69) * z + expr19 - expr73 * 
        a)/nu/expr27 + expr26 * expr69/expr60)/expr30
    grad["a"] = (((expr5 * expr3 + expr68 - expr84) * 
        z + ((2 * expr84 - expr73) * a + expr23))/nu/expr27 - 
        expr26 * expr84/expr60)/expr30 - expr28 * expr84/expr65
    grad["Ter"] = 1

    
    return (value, grad) if GRAD else value



def cvrt(nu, z, a, Ter=0, s=0.1, GRAD=False):
    """
    Compute exit (decision) time mean and variance _conditional_ on exit point (chosen
    alternative) of a diffusion process

    Arguments:

          nu: float. Drift rate.

           z: float. Starting point.

           a: float. Boundary separation

           s: Scaling parameter (Ratcliff's convention is ‘s = 0.1’, the
              default)
              
        GRAD: bool. Should the gradient be returned?

    see help(cmrt) for details.
    
    """
    import numpy as np
    from numpy import exp
    nu = np.array(nu)
    z = np.array(z)
    a = np.array(a)

    expr2 = -4 * nu
    expr4 = s**2
    expr5 = 2 * nu/expr4
    expr7 = exp(expr5 * a)
    expr8 = expr2 * expr7
    expr10 = 2 * z
    expr13 = exp(expr10 * nu/expr4)
    expr14 = -1 + expr13
    expr15 = expr8 * expr14
    expr17 = 4 * nu/expr4
    expr19 = exp(expr17 * a)
    expr20 = expr19 - expr13
    expr21 = expr15 * expr20
    expr22 = a**2
    expr25 = 2 * (z + a)
    expr28 = exp(expr25 * nu/expr4)
    expr29 = 4 * expr28
    expr30 = expr29 * nu
    expr31 = expr7 - 1
    expr32 = expr31**2
    expr33 = expr30 * expr32
    expr34 = z**2
    expr37 = 8 * expr28
    expr38 = expr37 * nu
    expr39 = expr38 * expr32
    expr40 = expr39 * a
    expr43 = 2 * expr4
    expr44 = expr43 * expr7
    expr45 = expr44 * expr14
    expr46 = expr45 * expr31
    expr48 = -expr7 + expr13
    expr49 = expr46 * expr48
    expr52 = expr4 * expr32
    expr54 = 4 * z
    expr57 = exp(expr54 * nu/expr4)
    expr58 = -expr19 + expr57
    expr59 = expr52 * expr58
    expr61 = expr21 * expr22 - expr33 * expr34 + expr40 * \
        z + expr49 * a - expr59 * z
    expr62 = expr61/expr32
    expr63 = nu**3
    expr64 = expr62/expr63
    expr65 = expr7 - expr13
    expr66 = expr65**2
    expr70 = expr7 * (2/expr4 * a)
    expr76 = expr13 * (expr10/expr4)
    expr82 = expr19 * (4/expr4 * a)
    expr88 = expr28 * (expr25/expr4)
    expr94 = 2 * (expr70 * expr31)
    expr132 = expr32**2
    expr147 = expr66**2
    expr150 = expr13 * expr5
    expr156 = expr28 * expr5
    expr159 = 4 * expr156 * nu * expr32
    expr166 = 8 * expr156 * nu * expr32
    expr191 = expr7 * expr5
    expr195 = expr19 * expr17
    expr203 = 2 * (expr191 * expr31)
    value = expr64/expr66

    m = np.size(value)
    grad = np.array([0 for i in range(m)], dtype=[('nu', 'float'),('z', 'float'),('a', 'float'),('Ter', 'float')])

    grad["nu"] = ((((((expr2 * expr70 - 4 * expr7) * expr14 + 
        expr8 * expr76) * expr20 + expr15 * (expr82 - expr76)) * 
        expr22 - ((4 * expr88 * nu + expr29) * expr32 + expr30 * 
        expr94) * expr34 + ((8 * expr88 * nu + expr37) * 
        expr32 + expr38 * expr94) * a * z + (((expr43 * expr70 * 
        expr14 + expr44 * expr76) * expr31 + expr45 * expr70) * 
        expr48 + expr46 * (expr76 - expr70)) * a - (expr4 * 
        expr94 * expr58 + expr52 * (expr57 * (expr54/expr4) - 
        expr82)) * z)/expr32 - expr61 * expr94/expr132)/expr63 - 
        expr62 * (3 * nu**2)/expr63**2)/expr66 - expr64 * (2 * 
        ((expr70 - expr76) * expr65))/expr147
    grad["z"] = ((expr8 * expr150 * expr20 - expr15 * 
        expr150) * expr22 - (expr159 * expr34 + expr33 * 
        expr10) + (expr166 * a * z + expr40) + (expr44 * 
        expr150 * expr31 * expr48 + expr46 * expr150) * 
        a - (expr52 * (expr57 * expr17) * z + expr59))/expr32/expr63/expr66 + \
        expr64 * (2 * (expr150 * expr65))/expr147
    grad["a"] = (((expr2 * expr191 * expr14 * expr20 + 
        expr15 * expr195) * expr22 + expr21 * (2 * a) - (expr159 + 
        expr30 * expr203) * expr34 + ((expr166 + expr38 * 
        expr203) * a + expr39) * z + (((expr43 * expr191 * 
        expr14 * expr31 + expr45 * expr191) * expr48 - expr46 * 
        expr191) * a + expr49) - (expr4 * expr203 * expr58 - 
        expr52 * expr195) * z)/expr32 - expr61 * expr203/expr132)/expr63/expr66 - \
        expr64 * (2 * (expr191 * expr65))/expr147
    grad["Ter"] = 0

    return (value, grad) if GRAD else value



def mrt(nu, z, a, Ter=0, s=0.1, GRAD=False):
    """
    Compute exit/decision time mean and variance irrespective of exit 
    point/chosen alternative

    Description:

         Given a boundary separation, a starting point, and a drift rate,
         this function computes the mean exit time/exit time variance of a
         one dimensional diffusion process under constant drift on an
         interval with absorbing boundaries. Used as a model of information
         accumulation, it is gives the mean decision time/decision time
         variance of responses in a speeded two-alternative forced choice
         (2AFC) response time task, regardless of whether the response is
         correct or incorrect.

    Usage:

           mrt(nu, z, a, s = 0.1, Ter)
           vrt(nu, z, a, s = 0.1)

    Arguments:

          nu: Drift rate. Float, list of floats, or numpy array.

           z: Starting point. Float, list of floats, numpy array.

           a: Boundary separation. Float, list of floats, numpy array.

           s: Scaling parameter (Ratcliff's convention is ‘s = 0.1’, the
              default)

         Ter: Lag (models non-decision time of a response time)
         
        GRAD: bool. Should the gradient be returned?

    Details:

         This model of information accumulation and decision is a
         simplified version of Ratcliff's diffusion model (1978). It can be
         used e.g., to compute the mean response times of the correct
         responses in a lexical decision time, given the drift rate, the
         bias (start point), and criterion (boundary separation).

    Value:

         ‘mrt’ returns the mean exit/decision time(s) ‘vrt’ returns
         the exit/decision time variance(s). The return value has the
         attribute "gradient" attached: the gradient with respect to each
         of the parameters.

    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘cmrt’, ‘cvrt’, ‘pe’, ‘batch’

    Examples:

         mrt( 0.1, 0.08, 0.12, Ter=0.25)
         vrt( 0.1, 0.08, 0.12)


    """
    import numpy as np
    from numpy import exp
    nu = np.array(nu)
    z = np.array(z)
    a = np.array(a)

    expr3 = s * s
    expr4 = -2 * nu/expr3
    expr6 = exp(expr4 * a)
    expr7 = expr6 - 1
    expr8 = a/expr7
    expr9 = expr8/nu
    expr11 = exp(expr4 * z)
    expr13 = 1/nu
    expr16 = expr13 * a
    expr20 = 2/expr3
    expr22 = expr6 * (expr20 * a)
    expr24 = expr7**2
    expr27 = nu**2
    expr35 = 1/expr27
    expr48 = expr6 * expr4
    value = expr9 * expr11 - expr13 * z - expr16/expr7 + \
        Ter

    m = np.size(value)
    grad = np.array([0 for i in range(m)], dtype=[('nu', 'float'),
        ('z', 'float'),('a', 'float'),('Ter', 'float')])

    grad["nu"] = (a * expr22/expr24/nu - expr8/expr27) * \
        expr11 - expr9 * (expr11 * (expr20 * z)) + expr35 * \
        z + (expr35 * a/expr7 - expr16 * expr22/expr24)
    grad["z"] = expr9 * (expr11 * expr4) - expr13
    grad["a"] = (1/expr7 - a * expr48/expr24)/nu * expr11 - \
        (expr13/expr7 - expr16 * expr48/expr24)
    grad["Ter"] = 1

    return (value, grad) if GRAD else value



def vrt(nu, z, a, Ter=0, s=0.1, GRAD=False):
    """
    Compute exit/decision time variance irrespective of exit point/chosen 
    alternative

    Arguments:

          nu: Drift rate. Float, list of floats, or numpy array.
           z: Starting point. Float, list of floats, numpy array.
           a: Boundary separation. Float, list of floats, numpy array.
           s: Scaling parameter (Ratcliff's convention is ‘s = 0.1’, the
              default)
         Ter: Lag (models non-decision time of a response time)

    See help(mrt) for details.

    """
    import numpy as np
    from numpy import exp
    nu = np.array(nu)
    z = np.array(z)
    a = np.array(a)

    expr4 = s**2
    expr5 = -2 * nu/expr4
    expr7 = exp(expr5 * z)
    expr8 = expr7 - 1
    expr9 = expr8**2
    expr10 = -nu * expr9
    expr11 = a**2
    expr13 = 4 * nu
    expr14 = expr13 * expr8
    expr16 = expr10 * expr11 - expr14 * expr11
    expr18 = exp(expr5 * a)
    expr19 = expr18 - 1
    expr20 = expr19**2
    expr23 = -3 * nu
    expr25 = expr13 * z
    expr26 = expr25 * a
    expr29 = expr23 * expr11 + expr26 + expr4 * a
    expr31 = expr29 * expr8 + expr26
    expr35 = expr16/expr20 + expr31/expr19 - expr4 * z
    expr36 = nu**3
    expr38 = 2/expr4
    expr40 = expr7 * (expr38 * z)
    expr53 = expr18 * (expr38 * a)
    expr57 = expr20**2
    expr61 = 4 * z * a
    expr80 = expr13 * a
    expr82 = expr7 * expr5
    expr98 = 2 * a
    expr103 = expr18 * expr5
    value = expr35/expr36

    m = np.size(value)
    grad = np.array([0 for i in range(m)], dtype=[('nu', 'float'),
        ('z', 'float'),('a', 'float'),('Ter', 'float')])

    grad["nu"] = (((nu * (2 * (expr40 * expr8)) - expr9) * 
        expr11 - (4 * expr8 - expr13 * expr40) * expr11)/expr20 + 
        expr16 * (2 * (expr53 * expr19))/expr57 + (((expr61 - 
        3 * expr11) * expr8 - expr29 * expr40 + expr61)/expr19 + 
        expr31 * expr53/expr20))/expr36 - expr35 * (3 * 
        nu**2)/expr36**2
    grad["z"] = ((expr80 * expr8 + expr29 * expr82 + 
        expr80)/expr19 - (nu * (2 * (expr82 * expr8)) * expr11 + 
        expr13 * expr82 * expr11)/expr20 - expr4)/expr36
    grad["a"] = ((expr10 * expr98 - expr14 * expr98)/expr20 - 
        expr16 * (2 * (expr103 * expr19))/expr57 + (((expr23 * 
        expr98 + expr25 + expr4) * expr8 + expr25)/expr19 - 
        expr31 * expr103/expr20))/expr36
    grad["Ter"] = 0

    return (value, grad) if GRAD else value



def pe(nu, z, a, Ter=0, s=0.1, GRAD=False):
    """
    Compute probability of exit through lower bound of a diffusion with
    constant drift

    Description:

         Computes the probability of exit through the lower bound of a
         univariate diffusion process with constant drift on an interval
         with absorbing boundaries. Used as a model of information
         accumulation, it is gives the probability of an error response in
         a speeded two-alternative forced choice (2AFC) response time task.

    Usage:

         pe(nu, z, a, s = 0.1)

    Arguments:

          nu: Drift rate. Float, list of floats, or numpy array.

           z: Starting point. Float, list of floats, numpy array.

           a: Boundary separation. Float, list of floats, numpy array

           s: Scaling parameter (Ratcliff's convention is ‘s = 0.1’, the
              default)

    Details:

         This process as a model of information accumulation and decision
         is Ratcliff's diffusion model (1978). It can be used e.g., to
         compute the mean response times of the correct responses in a
         lexical decision time, given the drift rate, the bias (start
         point), and criterion (boundary separation).

    Value:

         ‘pe’ returns the exit probability through lower end of the
         interval (0,a) The return value has the attribute "gradient"
         attached: the gradient with respect to each of the parameters.

    Author(s):

         Raoul P. P. P. Grasman    
         
    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘cmrt’, ‘cvrt’, ‘mrt’, ‘vrt’

    Examples:

         pe(.1, .08, .12)

     
    """
    import numpy as np
    from numpy import exp
    nu = np.array(nu)
    z = np.array(z)
    a = np.array(a)

    expr3 = s * s
    expr4 = -2 * nu/expr3
    expr6 = exp(expr4 * a)
    expr8 = exp(expr4 * z)
    expr9 = expr6 - expr8
    expr10 = expr6 - 1
    expr12 = 2/expr3
    expr14 = expr6 * (expr12 * a)
    expr20 = expr10**2
    expr27 = expr6 * expr4
    value = expr9/expr10
    
    m = np.size(value)
    grad = np.array([0 for i in range(m)], dtype=[('nu', 'float'),
        ('z', 'float'),('a', 'float'),('Ter', 'float')])

    grad["nu"] = -((expr14 - expr8 * (expr12 * z))/expr10 - 
        expr9 * expr14/expr20)
    grad["z"] = -(expr8 * expr4/expr10)
    grad["a"] = expr27/expr10 - expr9 * expr27/expr20
    grad["Ter"] = 0

    return (value, grad) if GRAD else value



def objf(par, par_names, ObsValPairs):
    """
    Objective function for finding the parameter values for solving the Method of Moments 
    equations for the EZ2 drift diffusion exit times. The objective function is simply the 
    squared sum of the deviations. This makes it possible to fit an underparameterized model 
    using least squares (i.e., generalized method of moments estimators).
    
    This function is intended to be private.
    
    See help(EZ2) for details.
    """
    import numpy as np
    p = dict(zip(par_names, par))
    
    objfval = 0.0
    objgrad = 0.0
    for observed, model_expr in ObsValPairs:
        predicted = eval(model_expr, globals(), p)
        error = predicted - observed
        objfval += 0.5 * np.sum(error**2) * 1e6
            
    return objfval



def pddexit(t, nu, z, a, top_boundary=True, joint_dist=False, s=0.1, tol = 1e-8, maxterms = 1000):
    """
    Cumulative distribution, density and quantile functions of exit times from 
    top or bottom boundary of a drift diffusion process.

    Description:

         Given a boundary separation, a starting point, and a drift rate,
         compute the probability F(t) = P(T<t), the density F'(t), the quantile
         F^-1(p), of the exit time T of a one dimensional diffusion process under 
         constant drift _nu_ on an interval (0, _a_) with absorbing boundaries,
         that started at _z_ inside (0, _a_). Either conditional on the point of 
         exit ('top' at _a_ or 'bottom' at 0), or regardless of the exit point. 

    Usage:

         pddexit(t, nu, z, a)
         dddexit(t, nu, z, a)
         qddexit(p, nu, z, a)

    Arguments:

           t: Float. Time of exit. 

           p: Float. Quantile.

          nu: Float. Drift rate.

           z: Float. Starting point.

           a: Float. Boundary separation.
           
        top_boundary: Bool. Exit point is top (True) or bottom (False) 
              boundary.
           
        joint_dist: Bool. ‘True’ evaluates the joint distribution 
              P(T < t, exit boundary is specified boundary). ‘False’
              (default) gives P(T < t | exit boundary is specified boundary).
              Specified boundary is boundary specified through ‘top_boundary’.
           
           s: Scaling parameter (default ‘s = 0.1’)
           
         tol: Float. Maximum tolerated precission error. 
         
        maxterms: Int. Maximum number of terms to evaluate in the infinite
              series (regardless of precision error).

    Details:
            A constant drift diffusion process can be characterized by the 
            Itô stochastic differential equation
            
            dX(t) = nu dt + s dW(t), X(0) = z
            
            with absorbing boundaries 0 and a. Here W(t) is the Wiener 
            process: W(0) = 0, W(t+dt)-W(t) ~ N(0, √dt), and W(t)-W(s) 
            is independent from W(s)-W(0) for all t > s > 0. 
            More intuitively it can be seen as the limit of a biased random 
            walk:
            
            S[n] ∝ ∑ (2*Bᵢ-1), Bᵢ ~ Bernoulli(p).

            The exit time (aka stoppint times, first passage times) of such 
            a process is the first time X(t) leaves the interval (0,a), after 
            which the process stops. The cumulative probability distribution 
            and density are computed from an infinite series. 
            
            For formulas, see references below.
            
    Value:

         A ‘numpy.ndarray’ of floats.

    Author(s):

         Raoul P. P. P. Grasman
         
    References:
             
         Cox, D. R., & Miller, H. D. (1970). _The theory of stochastic processes._
            London: Methuen.

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
            review_ vol. 85 (2) pp. 59-108.
         
         Tuerlinkckx, F. (2004). The efficient computation of the cumulative 
            distribution and probability density functions in the diffusion model. 
            _Behavior Research Methods, Instruments, & Computers, 36_(4), 702-716.

    See Also:

         ‘EZ2-package’, ‘rddexitj’, ‘ddexit_fit’

    Examples:

             pddexit([i/10 for i in range(1,9)], .12, 0.08, 0.12)

    """
    import numpy as np
    from numpy import exp, sin, pi
    nu, z, a = (nu/s, z/s, a/s) if not top_boundary else (-nu/s, (a-z)/s, a/s)
    S = 0
    t = np.array(t)
    tmin = t.min()
    for k in range(1, maxterms):
        
        term_without_sin = 2 * k * exp(-0.5 * (pi*k/a)**2 * t) / (nu**2 + (pi*k/a)**2)
        S += term_without_sin * sin(z*pi*k / a) 

        max_term_without_sin = 2 * k * exp(-0.5 * (pi*k/a)**2 * tmin) / (nu**2 + (pi*k/a)**2)
        if np.abs(max_term_without_sin) < tol:
            break
    
    P0 = (exp(-2*a*nu) - exp(-2*z*nu)) / (exp(-2*a*nu) - 1)
    P = P0 - (pi/a**2) * exp(-z*nu) * exp(-.5*nu**2*t) * S

    return np.maximum(P,0) if joint_dist else np.maximum(P / P0, 0)



def dddexit(t, nu, z, a, top_boundary=True, joint_dens=False, s=0.1, tol = 1e-8, maxterms = 1000, debug=False):
    """
    Compute the density of exit times from top or bottom boundary of a drift 
    diffusion process.

    See ‘help(pddexit)’ for complete help.
            
    Examples:

             pddexit([0.25, .5, .75], nu=0.1, z=0.08, a=0.12)
    """
    import numpy as np
    from numpy import exp, sin, pi
    nu, z, a = (nu/s, z/s, a/s) if not top_boundary else (-nu/s, (a-z)/s, a/s)
    S, dS = 0, 0
    t = np.array(t)
    tmin = t.min()
    for k in range(1, maxterms):
        
        term_without_sin = 2 * k * exp(-0.5 * (pi*k/a)**2 * t) / (nu**2 + (pi*k/a)**2)
        dterm_without_sin = -0.5 * (pi*k / a)**2 * term_without_sin
        S  +=  term_without_sin * sin(z*pi*k / a) 
        dS += dterm_without_sin * sin(z*pi*k / a) 

        max_term_without_sin  = 2 * k * exp(-0.5 * (pi*k/a)**2 * tmin) / (nu**2 + (pi*k/a)**2)
        dmax_term_without_sin = -0.5 * (pi*k / a)**2 * max_term_without_sin
        if np.abs(max_term_without_sin) < tol and np.abs(dmax_term_without_sin) < tol:
            break
    
    if debug:
        print(f"nr of terms: {k}\nlast term smaller than {min(max_term_without_sin,max_term_without_sin)}")
    
    P0 = (exp(-2*a*nu) - exp(-2*z*nu)) / (exp(-2*a*nu) - 1)
    P = P0 - (pi/a**2) * exp(-z*nu) * exp(-.5*nu**2*t) * S
    dU_S = 0.5*(pi*(nu/a)**2) * exp(-z*nu) * exp(-.5*nu**2*t) * S
    U_dS = - (pi/a**2) * exp(-z*nu) * exp(-.5*nu**2*t) * dS
    dP = dU_S + U_dS

    return np.maximum(dP,0) if joint_dens else np.maximum(dP / P0, 0)



def qddexit(p, nu, z, a, top_boundary=True, joint_dens=False, tol=1e-8, *kwargs):
    """
    Compute the quantiles for the cumulative distribution function of 
    exit times from top or bottom boundary of a drift diffusion process.

    See ‘help(pddexit)’ for complete help.
    
    Details:
    
        This function uses the bisection root finding method for inverting
        the cumulative distribution function ‘pddexit’.
        
    Examples:

             qddexit([0.25, .5, .75], nu=0.1, z=0.08, a=0.12)
    """
    import numpy as np
    
    pv = np.array([p]) if not hasattr(p, "__len__") else np.array(p)
    
    # Solve 
    #   p = pddexit(x, nu, z, a, top_boundary=True) 
    # by means of bisection
    
    largest_likely_value = .01
    n = len(pv)
    while pddexit(largest_likely_value, nu, z, a, top_boundary, *kwargs) < 1 - 0.01/n:
        largest_likely_value += 1
        
    L = (0.01/n) * np.ones(n)
    R = largest_likely_value * np.ones(n)
    pL = pddexit(L, nu, z, a, top_boundary) 
    pR = pddexit(R, nu, z, a, top_boundary) 
    
    for _ in range(1000):
        c = (L + R) / 2
        pC = pddexit(c, nu, z, a, top_boundary, *kwargs) 
        if np.abs(pC).max() < tol or np.abs(R-L).max() < tol:
            break
        L  = np.where(pC  < pv, c, L)
        R  = np.where(pC >= pv, c, R)
        pL = np.where(pC  < pv, pC, pL)
        pR = np.where(pC >= pv, pC, pR)
        
    return c if len(c) > 1 else c[0]



def rddexit(size, nu, z, a, top_boundary=True):
    """
    Generate random sample of exit times from top or bottom boundary of a 
    drift diffusion process.

    Description:

         Given a boundary separation, a starting point, and a drift rate,
         this function generates exit times of a one dimensional diffusion 
         process under constant drift on an interval with absorbing 
         boundaries, conditioned on the point of exit. 

    Usage:

         rddexit(size, nu, z, a)

    Arguments:

         size: Number of samples drawn.

           nu: Drift rate.

            z: Starting point.

            a: Boundary separation

    Details:

            This function uses the quantile function (inverse cumulative 
            distribution function) method.
            
    Value:

         A numpy.array of floats (the exit times from the boundary 
         specified through `top_boundary`).

    Author(s):

         Raoul P. P. P. Grasman

    See Also:

         ‘EZ2-package’, ‘rddexitj’, ‘pddexit’, ‘qddexit’, ‘dddexit’

    Examples:

             rddexit( 0.1, 0.08, 0.12)

    """
    import numpy as np
    
    n = size
    
    # uniform random sample from [0,1)
    pv = np.random.random_sample(n)
    c = qddexit(pv, nu, z, a, top_boundary)
    
    ### The following was moved to `qddexit`
    ## solve pv = pddexit(x, nu, z, a, top_boundary=True) by means of bisection
    #largest_likely_value = .01
    #while pddexit(largest_likely_value, nu, z, a) < 1 - 0.01/n:
    #    largest_likely_value += 1
    #    
    #L = (0.01/n) * np.ones(n)
    #R = largest_likely_value * np.ones(n)
    #pL = pddexit(L, nu, z, a, top_boundary) 
    #pR = pddexit(R, nu, z, a, top_boundary) 
    #
    #for _ in range(1000):
    #    c = (L + R) / 2
    #    pC = pddexit(c, nu, z, a, top_boundary) 
    #    if np.abs(pC).max() < tol or np.abs(R-L).max() < tol:
    #        break
    #    L  = np.where(pC  < pv, c, L)
    #    R  = np.where(pC >= pv, c, R)
    #    pL = np.where(pC  < pv, pC, pL)
    #    pR = np.where(pC >= pv, pC, pR)
        
    return c 



def rddexitj(size, nu, z, a, s=0.1):
    """
    Generate random sample of exit times from top and bottom boundaries of a 
    drift diffusion process.

    Description:

         Given a boundary separation, a starting point, and a drift rate,
         this function generates exit times of a one dimensional diffusion 
         process under constant drift on an interval with absorbing 
         boundaries, conditioned on the point of exit. 

    Usage:

         rddexitj(size, nu, z, a)

    Arguments:

         size: Number of samples drawn.

           nu: Drift rate.

            z: Starting point.

            a: Boundary separation

    Details:

            This function uses the quantile function (inverse cumulative 
            distribution function) method.
            
    Value:

         A 2-tuple with components:
         
             rt: list of two numpy.array of floats: 
                 rt[0] are exit times at the bottom boundary; rt[1]
                 are exit times at the top boundary.
                 
             pe: float. Theoretical probability of exit at the bottom.

    Author(s):

         Raoul P. P. P. Grasman

    See Also:

         ‘EZ2-package’, ‘rddexitj’, ‘pddexit’, ‘qddexit’, ‘dddexit’

    Examples:

             rddexit( 0.1, 0.08, 0.12)

    """
    import numpy as np
    from numpy import pi, exp
    
    n = size
    s2 = s*s
    P0 = (exp(-2*a*nu/s2) - exp(-2*z*nu/s2)) / (exp(-2*a*nu/s2) - 1)
    n_bottom = np.random.binomial(n, P0)
    n_top = n - n_bottom
    
    et_0 = list(rddexit(n_bottom, nu, z, a, top_boundary=False)) if n_bottom > 0 else []
    et_a = list(rddexit(n_top, nu, z, a, top_boundary=True)) if n_top > 0 else []
    
    return (et_0, et_a), P0



def ez2_2afc_(vrt_w, pe_w, vrt_nw, pe_nw, start=None, correct_only=True, 
      method='lm', *kwargs):
    """
    Fit simple diffusion model to observed sample moments of 2AFC task responses.

    Description:

         Fit the a simplified diffusion model for response time and
         accuracy to observed proportions of errors and response time means
         and variances in speeded two-alternative forced choice (2AFC) paradigms.

         A prototypical example is a lexical decision task: words and non-words
         are displayed and participants have to indicate whether it is an existing
         word. Response times (RTs) and response correctness are recorded for
         both words and non-words over multiple trials for each.

    Usage:

         ez2_2afc_(vrt_w, pe_w, vrt_nw, pe_nw)

    Arguments:

       vrt_w: float. Observed RT variance for type 'w' stimuli.

        pe_w: float. Observed % of error responses for type 'w' stimuli.

      vrt_nw: float. Observed RT variance for type 'nw' stimuli.

       pe_nw: float. Observed % of error responses for type 'w' stimuli.

       start: list. Optional list of starting values for the parameters. If
              not provided, starting values are derived from 'data2ez'.

correct_only: bool. If True (default) ‘vrt_w’ and ‘vrt_nw’ are assumed to
              be computed from correct responses only. If False ‘vrt_w’
              and ‘vrt_nw’ are assumed to be computed from response RTs
              regardless of response correctness.
              
      method: str. See ‘scipy.optimize.root’ for details.

     *kwargs: Further optional arguments to ‘scipy.optimize.root’.

    Details:

         ez2_2afc_ fits a simplified version of the diffusion model for 
         human and animal response times and response accuracy to moments
         (means, variances, proportions of errors) of observed responses.

         Fitting is done using ‘scipy.optimize.root’ for solving the 
         moments' equations

          vrt_w = EZ_vrt(nu1, z, a)
           pe_w = EZ_pe(nu1, z, a)
         vrt_nw = EZ_vrt(nu2, a-z, a)
          pe_nw = EZ_pe(nu2, a-z, a)

        Estimators are therefore 'Method of Moments' estimators.

        The parameters ‘nu1’ and ‘nu2’ reflect the difficulty of classifying 
        the stimulus, for stimulus type 'w' and 'nw' respectively; ‘z’ 
        reflects a participant's response bias; and ‘a’ reflect a participant's 
        response cautiousness.

        Use ‘ez2_2afc’ for fitting multiple sets of data.

        Use 'EZ2' or 'batch' for fitting more general models.

    Value:

         The object returned by optim. This object has attributes

           x: A list with the estimated parameter values [nu1,nu2,z,a]

         For other list members, see ‘Value’ section of ‘scipy.optimize.root’ 
         for a complete description.

    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘ez2_2afc’, ‘EZ2’, ‘batch’

    Examples:

            nu1, nu2, Z, A = .12, .135, .15, .22
            vrt0, pe0, vrt1, pe1 = [
                vrt(nu1, Z, A), 
                pe(nu1, Z, A), 
                vrt(nu2, A-Z, A), 
                pe(nu2, A-Z, A)
            ]

            ez2_2afc_(vrt0, pe0, vrt1, pe1)

    """
    import numpy as np
    from scipy import optimize
    
    assert start == None or len(start) == 4, "'start' should be None or contain 4 values"
    
    if start == None:
        v_w,  a_w,  _ = data2ez(1-pe_w,  vrt_w )
        v_nw, a_nw, _ = data2ez(1-pe_nw, vrt_nw)
        start = [v_w, v_nw, (a_w+a_nw)/4, (a_w+a_nw)/2]
    
    obs = [vrt_w, pe_w, vrt_nw, pe_nw]
    def objf(x):
        var_func = cvrt if correct_only else vrt
        v1, v2, z, a = x
        f = [var_func(v1, z, a) - vrt_w, pe(v1, z, a) - pe_w, 
             var_func(v2, a-z, a) - vrt_nw, pe(v2, a-z, a) - pe_nw]
        return f

    sol = optimize.root(objf, start, jac=False, method=method, *kwargs)

    return sol



def ez2_2afc(vrt_w, pe_w, vrt_nw=None, pe_nw=None, start=None, correct_only=True, 
             method='lm', return_convergence=False, *kwargs):
    """
    Fit simple diffusion model to observed sample moments of 2AFC task responses.

    Description:

         Wrapper around ‘ez2_2afc_’ for fitting multiple cases. See
         ‘help(ez2_2afc_)’ for details.

    Usage:

         ez2_2afc(vrt_w, pe_w, vrt_nw, pe_nw)

    Arguments:

       vrt_w: list like or data frame. List of RT variances for type 
              'w' stimuli, or pandas.DataFrame containing columns that
              contain ‘vrt_w’, ‘pe_w’, ‘vrt_nw’, and ‘vrt_nw’.

        pe_w: list like. If ‘vrt_w’ is a list of RT variances, a list of 
              observed % of error responses for type 'w' stimuli of the
              same length as ‘vrt_w’. Or, if ‘vrt_w’ is a pandas.DataFrame, 
              a list of column names of columns in ‘vrt_w’ that correspond
              to ‘vrt_w’, ‘pe_w’, ‘vrt_nw’, and ‘vrt_nw’.

      vrt_nw: list like. Observed RT variances for type 'nw' stimuli. 
              Ignored if ‘vrt_w’ is a pandas.DataFrame.

       pe_nw: float. Observed %'s of error responses for type 'w' stimuli.
              Ignored if ‘vrt_w’ is a pandas.DataFrame.

       start: list. Optional list of starting values for the parameters. If
              not provided, starting values are derived from 'data2ez'.
              
correct_only: bool. If True (default) ‘vrt_w’ and ‘vrt_nw’ are assumed to
              be computed from correct responses only. If False ‘vrt_w’
              and ‘vrt_nw’ are assumed to be computed from response RTs
              regardless of response correctness.

      method: str. See ‘scipy.optimize.root’ for details.
      
return_convergence: bool. If True, all ‘scipy.optimize.root’ information
              is returned in addition to parameter estimates.

     *kwargs: Further optional arguments to ‘scipy.optimize.root’.

    Details:

         ‘ez2_2afc’ iteratively calls ‘ez2_2afc_’ to estimate the parameters
        ‘nu1’, ‘nu2’, ‘z’, ‘a’ for each of the cases.

        Use 'EZ2' or 'batch' for fitting more general models.

    Value:

         A pandas.DataFrame with the estimated parameters for each case if
         ‘return_convergence’ = False. If ‘return_convergence’ = True a
         2-tuple is returned with two pandas.DataFrame-s: The first
         is the data frame with estimates; the second is a data frame
         with the convergence information returned by ‘ez2_2afc_’.

    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘ez2_2afc_’, ‘EZ2’, ‘batch’

    Examples:

        import pandas as pd
        from EZ2 import vrt, pe, ez2_2afc

         ## Create some data (theoretical values, not simulated) for a typic
         ## 2AFC experiment — in this example a lexical decision task
         ## (Needless to say, in reality you would moments computed from real data!)

         # true parameter values (10 different cases)
         par_df = pd.DataFrame({
             "v0": [0.1 + (0.3-0.1)*i/10 for i in range(11)],
             "v1": [0.15 + (0.4-0.15)*i/10 for i in range(11)],
             "z":  [0.15 + 0.03*(i-5)/5 for i in range(11)],
             "a":  [0.25]*11
         })

         # compute the theoretical variance (vrt0) and proportion error (pe0) for 
         # the 'word' response times, and the theoretical variance (vrt1) and error
         # (pe1) for the 'non-word' response times.
         dat_df = pd.DataFrame({
            'vrt0': eval('cvrt(v0,z,a)', globals(), par_df),
            'pe0' : eval('pe(v0,z,a)', globals(), par_df),
            'vrt1': eval('cvrt(v1,a-z,a)', globals(), par_df),
            'pe1' : eval('pe(v1,a-z,a)', globals(), par_df)
         })
        dat_df         # pretend that `dat_df` is the data frame that 
                       # was computed from real data; each row containing
                       # vrt0, pe0, vrt1, and pe1 from a single participant

        ## recover the parameters from the pretend data `dat_df`

        # call with lists of values
        ez2_2afc(dat_df.vrt0.values, dat_df.pe0.values, dat_df.vrt1.values, dat_df.pe1.values)

        # call pandas data frame and column names 
        # (names can be anything, but the order of names is important!)
        ez2_2afc(dat_df, ["vrt0", "pe0", "vrt1", "pe1"])

    """
    import pandas
    
    if type(vrt_w) == pandas.core.frame.DataFrame:
        
        assert all(type(w) == str and w in vrt_w for w in list(pe_w)), \
            "Second argument should be list of exiting column names when the first is a data frame."
        
        # extract the indicated columns
        dat_df, cnames = vrt_w, pe_w
        vrt_w, pe_w, vrt_nw, pe_nw = list(dat_df[cnames].to_numpy().T)
        
    #assert not isinstance(vrt_nw, None) and not isinstance(pe_nw, None), \
    #    "'vrt_nw' and 'pe_nw' should be assigned if first argument is not a data frame"
    
    vrt_w  = [vrt_w]  if not hasattr(vrt_w, "__len__")  else vrt_w
    pe_w   = [pe_w]   if not hasattr(pe_w, "__len__")   else pe_w
    vrt_nw = [vrt_nw] if not hasattr(vrt_nw, "__len__") else vrt_nw
    pe_nw  = [pe_nw]  if not hasattr(pe_nw, "__len__")  else pe_nw
    
    assert len(vrt_w) == len(pe_w) == len(vrt_nw) == len(pe_w), \
        "'vrt_w', 'pe_w', 'vrt_nw', 'pe_nw' should all have equal length"
    
    sols, res = [], []
    for i in range(len(vrt_w)):
        sol = ez2_2afc_(vrt_w[i], pe_w[i], vrt_nw[i], pe_nw[i], start, correct_only, method, *kwargs)
        sols.append(sol.x)
        res.append(dict(sol))
    
    estimates = pandas.DataFrame(sols, columns=['nu1','nu2','z','a'])
    info = pandas.DataFrame(res)
    return (estimates, info) if return_convergence else estimates 



def ddexit_fit(rt, start, top_boundary=True, offset=False, method=None, suppress_warnings=True, *kwargs):
    """
    Maximum likelihood estimation of parameters nu, z, a (and optionally an offset) 
    of a constant drift diffusion process from the exit times of hitting either or both
    bounds.
    
    See help(pddexit) for detailed information on the parameters.
    
    Usage:
    
        ddexit_fit(rt, start)
        
    Input:
    
        rt: List like. If len(rt) = 2 then rt[0] is assumed to be a list of 
            exit times at the bottom boundary, and rt[1] is assumed to be a 
            list of exit times at the top boundary. Otherwise rt is a list
            of exit times either at the top or bottom boundary (specified
            through `top_boundary`.
            
      top_boundary: Bool. True if 'rt' contains exit times at top boundary.
            False if 'rt' contains exit times at bottom boundary.
            
        offset: Bool. Should an additive offset to the exit times be fitted?
        
        method: None or string. Passed onto 'scipy.optimize.minimize'.
        
      *kwargs: Further optional arguments passed on to 
           'scipy.optimize.minimize'. See help(scipy.optimize.minimize).
           
    Value:
    
        Object returned by 'scipy.optimize.minimize'. 
        
        Attributes:
        
                 x: Estimated parameters.
            
          hess_inv: Inverse of Hessian matrix at minimum of the negative
                    log-likelihood function if method = 'BFGS'. This
                    matrix is an approximation to the variance covariance
                    matrix of the estimates.
                    
               ...: See help('scipy.optimize.minimize') for details.
               
    Author(s):
    
        R. P. P. P. Grasman <r.p.p.p.grasman@uva.nl>
        
    Examples:
    
            np.random.seed(29)
            rt = rddexitj(50, nu, z, a)[0] # 2-tuple of bottom and top exit times
            ddexit_fit(rt, [.1, .1, .2])
            
            # exit times with offset
            np.random.seed(101)
            rt = rddexitj(5000, nu, z, a)[0]
            rt = [np.array(rt[0]) + 0.33, np.array(rt[1]) + 0.33]
            ddexit_fit(rt, [.2,.1,.17,.1], offset=True)
            
            # use Nelder-Mead (sometimes more stable, but no Hessian)
            ddexit_fit(rt, [.2,.1,.17,.1], offset=True, method='Nelder-Mead')
    """
    import numpy as np
    from scipy import optimize as optim
    import warnings
    
    assert hasattr(start,"__len__") and len(start) == 3 if not offset else len(start) == 4, \
        "'start' should have 3 (no offset) or 4 (w/ offset) values"
    
    if len(rt) == 2:
        if offset:
            nll = lambda x: -2.0 * np.log(dddexit(np.array(rt[0])-x[3], x[0], x[1], x[2], False)).sum() - \
                                2.0 * np.log(dddexit(np.array(rt[1])-x[3], x[0], x[1], x[2], True)).sum()
        else:
            nll = lambda x: -2.0 * np.log(dddexit(rt[0], x[0], x[1], x[2], False)).sum() - \
                                2.0 * np.log(dddexit(rt[1], x[0], x[1], x[2], True)).sum()
    else:
        if offset:
            nll = lambda x: -2.0 * np.log(dddexit(np.array(rt[0])-x[3], x[0], x[1], x[2], top_boundary)).sum()
        else:
            nll = lambda x: -2.0 * np.log(dddexit(np.array(rt[0]), x[0], x[1], x[2], top_boundary)).sum()


    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")
        res = optim.minimize(nll, start) if method==None else optim.minimize(nll, start, method=method, *kwargs)
        
    return res



def data2ez(Pc, VRT, MRT = None, s = 0.1):
    """Convert observed sample moments to diffusion parameter values of
       a drift diffusion process with absorbing boundaries 0 and a,
       starting at a/2.

    Description:

         Converts proportion correct (Pc), response time variance (VRT),
         and response time mean (MRT), to EZ diffusion model parameters.

    Usage:

         data2ez(Pc, VRT, MRT, s = 0.1)

    Arguments:

          Pc: Proportion of correct responses.

         VRT: Variance of the response times in seconds.

         MRT: Mean of the response times in seconds.

           s: Scale parameter for the parameters (Ratcliff's convention is
              ‘s = 0.1’, the default).
              
    Details:

         Use of MRT is optional; if MRT is absent ‘Ter’ will not be
         computed.

    Value:

         A tuple with estimates

          v : Drift rate

          a : Boundary separation

         Ter: Non-decision time (is None if MRT is absent)

    Author(s):

         Raoul Grasman
         
    References:

         Grasman, R.P.P.P., Wagenmakers, E.-J.& van der Maas, H.L.J.
         (2009). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation.
         Journal of Mathematical Psychology, 53(2), 55-68.

    See Also:

         ‘batch’

    Examples:

         data2ez(Pc=0.8022, VRT=0.112035, MRT=0.7231)
         
     """
    import numpy as np
    from numpy import exp, log, sign
    Pc = np.array(Pc)
    VRT = np.array(VRT)
    if MRT != None:
        MRT = np.array(MRT)
    logit = lambda p: log(p/(1 - p))
    
    p = Pc
    if np.any(p == 0):
        raise ValueError("Oops, only error responses!")
    if np.any(p == 0.5):
        raise ValueError("Oops, chance performance!")
    if np.any(p == 1):
        raise ValueError("Oops, only correct responses!")

    s2 = s * s
    L = logit(p)
    x = L * (L * p * p - L * p + p - 0.5)/VRT
    v = sign(p - 0.5) * s * pow(x, (1/4))
    a = s2 * logit(p)/v
    y = -v * a/s2
    
    MDT = (a/(2 * v)) * (1 - exp(y))/(1 + exp(y))
    
    Ter = MDT * 0.0
    Ter = None if np.any(MRT == None) else  MRT - MDT
    return v, a, Ter



def v1(p_start, ObsValPairs, **kwargs):
    from scipy import optimize
    """
    Fit diffusion model to observed sample moments using generalized
    method of moments.

    Description:

         Fit the a simplified diffusion model for response time and
         accuracy to observed proportions of errors and response time means
         and variances.

    Usage:

         v1(pstart, ObsValPair, ...)

    Arguments:

      pstart: Dictionary with named starting parameter values

    ObsValPair: List of observed-predicted value pair in the form (0.80,
              "pe(v1, z, a)") or (vrt2, "pe(v2, a-z, a)") if `vrt2`
              exists in the global environment and is numeric.

         ...: More arguments that are passed to `scipy.optimize.minimize`.

    Details:

         EZ2 fits a simplified version of the diffusion model for human and
         monkey response times and accuracy to the means and variances of
         the observables. This model of information accumulation and
         decision is a simplified version of Ratcliff's diffusion model
         (1978).

         Use ‘batch’ for more user friendly fitting automatically each
         row in a ‘pandas.DataFrame’.

    Value:

         The 2-tuple which contains

         par: A dictionary containing the parameter estimates
         
         fit: OptimizeResult; the object returned by scipy.optimize.minimize. 
            It has attributes:

             fun: Sum of squared prediction errors. This should be very close
                  to zero (order of ‘1e-8’) if there are as many
                  observed-predicted moment value pairs as there are unknown
                  parameters (the estimates then constitute method of moments
                  estimators).
                  
             succes: bool. Did the algorithm converge as expected. If this
                 is False, this doesn't necesarily mean the parameters are wrong.
                 When the number of observed moments is equal to the number of
                 parameters this is frequently the case, even if the parameters
                 are correct.
                 
             For other object content, see ‘Value’ section of ‘optim’ for a
             complete description.

    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘batch_v1’, ‘batch’, ‘EZ2’

    Examples:

        # An example with theoretical population values as 'observations'

        # generate observations:
        Vrt0 = vrt(0.1, 0.08, 0.12)        # RT variance 'word' responses
        Pe0  = pe(0.1, 0.08, 0.12)         # Proportion errors 'word' response 
        Vrt1 = vrt(0.15, 0.12-0.08, 0.12)  # RT variance 'non-word' responses
        Pe1  = pe(0.15, 0.12-0.08, 0.12)   # Proportion errors 'non-word' responses

        # specify the model for the 'observations': "v0", "v1", "z", and "a" are
        # the names of the model parameters to be estimated by EZ2
        ObsValPairs = [
            (Vrt0, "vrt(v0,z,a)"),
            (Pe0,  "pe(v0,z,a)"),
            (Vrt1, "vrt(v1,a-z,a)"),
            (Pe1,  "pe(v1, a-z, a)")
        ]

        # find the model parameters using an arbitrary starting point:
        EZ2({"v0": .156, "v1" :.1144, "z":.0631, "a": .1263}, ObsValPairs)

        # the result should be: "v0" ≈ 0.1, "v1" ≈ 0.15, "z" ≈ 0.08, and "a" ≈ 0.12
    """
    import numpy as np

    par = list(p_start.values())
    par_names = list(p_start.keys())
    fit = optimize.minimize(objf, x0=par, args=(par_names, ObsValPairs), **kwargs)
    
    p_end = dict(zip(list(p_start.keys()), list(fit.x)))
    
    return p_end, fit



def batch_v1(pstart, column_models, data, nrestart = 4, **kwargs):
    """
    Batch EZ2 model fitting using generalized method of moments.

    Description:

         Fit the a simplified diffusion model for response time and
         accuracy in batch to each row of a data frame.

    Usage:

         batch_v1(pstart, column_models, data, nrestart = 4, **kwargs)

    Arguments:

      pstart: Dictionary with model parameter starting values (floats).
              (Starting values are good guesses of the true value of the 
              parameters.)

    column_models: List of len(column_models) = len(data.columns) expressions 
              as strings that express the observed values of the means, variances, 
              proportions in the columns of data in terms of model parameters. 
              See examples below.


        data: Data frame with observed values of moments (RT means, 
              variances, proportions response errors) for different
              experimental conditions.

    nrestart: Number of times the fitting should restarted. Defaults to 4.

         ...: Named arguments passed onto scipy.optimize.minimize.

    Details:

         For each row in `data` the system of Method of Moments equations 

             value in `data` column j = value of expression in `column_models[j]`

         are solved for the model parameters used in the `column_models` 
         expressions and defined in `pstart`. 

         Solving is achieved by minimizing the _total discrepancy_ between the 
         values in the columns of `data` and the values predicted by the 
         expressions in `column_models`.

         The number of restarts needs to be increased if the model entails
         the same number of estimated parameters as the number of modeled
         moments and any of the values in the ‘fval’ column in the
         returned data frame is not close to ‘1.0e-08’.

    Value:

         DataFrame with parameter estimates for the column models fitted
         to each row in `data`. In addition, convergence information 
         about the solving the equations is given for each row:

           fval: The value of the total discrepancy between values in 
                `data` and the value predicted from the model expression.

          niter: The number of iterations taken by the solving algorithm.

         status: Integer that reports the convergence state of the 
                 algorithm. See help(scipy.optimize.OptimizeResult) 
                 for details. 

         succes: Bool. True implies succesful convergence. A very small
                 `fval` is much more important than `status` = True.

         norm_jac: Euclidean norm of the gradient. Should be 0 in theory.
                 Values larger than 1.0 may indicate improper solution
                 for that row, especially with non-small values of `fval`.
                 Try to recompute such cases for better solutions.


    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘v1’, ‘EZ2’, ‘batch’

    Examples:

         import pandas as pd

         ## create some data (theoretical values, not simulated) for a typic
         ## 2AFC experiment — in this example a lexical decision task
         ## (Needless to say, in reality you would moments computed from real data!)

         # true parameter values (10 different cases)
         par_df = pd.DataFrame({
             "v0": [0.1 + (0.3-0.1)*i/10 for i in range(11)],
             "v1": [0.15 + (0.4-0.15)*i/10 for i in range(11)],
             "z":  [0.15 + 0.03*(i-5)/5 for i in range(11)],
             "a":  [0.25]*11
         })

         # compute the theoretical variance (vrt0) and proportion error (pe0) for 
         # the 'word' response times, and the theoretical variance (vrt1) and error
         # (pe1) for the 'non-word' response times.
         dat_df = pd.DataFrame({
            'vrt0': eval('vrt(v0,z,a)', globals(), par_df),
            'pe0' : eval('pe(v0,z,a)', globals(), par_df),
            'vrt1': eval('vrt(v1,a-z,a)', globals(), par_df),
            'pe1' : eval('pe(v1,a-z,a)', globals(), par_df)
         })
         dat_df          # now pretend that `dat_df` is the data frame that 
                         # you have computed from real data; each row containing
                         # vrt0, pe0, vrt1, and pe1 from a single participant

         ## recover the parameters from the pretend data `dat_df`

         # specify the model expressions for each column
         column_models = [
             'vrt(v0,z,a)',        # first column: vrt0
             'pe(v0,z,a)',         # second column: pe0
             'vrt(v1,a-z,a)',      # third column: vrt1, starting point = a-z
             'pe(v1, a-z, a)']     # fourth column: pe1

         # solve for parameters: try 16 random starting values for each row
         pstart = {'v0': 0.17, 'v1': 0.15, 'z': 0.12, 'a': 0.25}

         random.seed(11)
         ez2fit = batch(pstart, column_model, dat_df, nrestart=16, tol=1e-15)
         ez2fit

    """

    from random import uniform
    import warnings
    import pandas as pd

    # pstart, column_models, data, nrestart = 4
    dat_df = data
    res = []
    resfit = []
    for i in range(len(dat_df)):
        
        ObsValPair = list(zip(dat_df.loc[i], column_models))
        p_end = pstart
        fval = 1e6
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for _ in range(nrestart):
                
                p_start = {parname: value * (1+uniform(-0.1,0.1)) for parname, value in p_end.items()}
                p_tmp, fit = v1(p_start, ObsValPair, **kwargs)
                
                if fit.fun < fval:
                    p_end = p_tmp
                    fval = fit.fun
                    
            
            fit_diagn = {"fval":fit.fun, "niter":fit.nit, "status": fit.status, 
                         "success": fit.success, "norm_jac": sum(fit.jac**2,0)**0.5}
            p_end.update(fit_diagn)
        
            res.append(p_end)
        
    return pd.DataFrame(res) 



def EZ2(p_start, ObsValPairs, **kwargs):
    """
    Fit diffusion model to observed sample moments

    Description:

         Fit the a simplified diffusion model for response time and
         accuracy to observed proportions of errors and response time means
         and variances.

    Usage:

         EZ2(pstart, ObsValPair, ...)

    Arguments:

      pstart: Dictionary with named starting parameter values

    ObsValPair: List of observed-predicted value pair in the form (0.80,
              "pe(v1, z, a)") or (vrt2, "pe(v2, a-z, a)") if `vrt2`
              exists in the global environment and is numeric.

         ...: More arguments that are passed to `scipy.optimize.minimize`.

    Details:

         EZ2 fits a simplified version of the diffusion model for human and
         monkey response times and accuracy to the means and variances of
         the observables. This model of information accumulation and
         decision is a simplified version of Ratcliff's diffusion model
         (1978).

         Use ‘batch’ for more user friendly fitting automatically each
         row in a ‘pandas.DataFrame’.

    Value:

         The 2-tuple which contains

         par: A dictionary containing the parameter estimates
         
         fit: OptimizeResult; the object returned by scipy.optimize.minimize. 
            It has attributes:

             fun: Sum of squared prediction errors. This should be very close
                  to zero (order of ‘1e-8’) if there are as many
                  observed-predicted moment value pairs as there are unknown
                  parameters (the estimates then constitute method of moments
                  estimators).
                  
             succes: bool. Did the algorithm converge as expected. If this
                 is False, this doesn't necesarily mean the parameters are wrong.
                 When the number of observed moments is equal to the number of
                 parameters this is frequently the case, even if the parameters
                 are correct.
                 
             For other object content, see ‘Value’ section of ‘optim’ for a
             complete description.

    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘EZ2-package’, ‘batch’

    Examples:

        # An example with theoretical population values as 'observations'

        # generate observations:
        Vrt0 = vrt(0.1, 0.08, 0.12)        # RT variance 'word' responses
        Pe0  = pe(0.1, 0.08, 0.12)         # Proportion errors 'word' response 
        Vrt1 = vrt(0.15, 0.12-0.08, 0.12)  # RT variance 'non-word' responses
        Pe1  = pe(0.15, 0.12-0.08, 0.12)   # Proportion errors 'non-word' responses

        # specify the model for the 'observations': "v0", "v1", "z", and "a" are
        # the names of the model parameters to be estimated by EZ2
        ObsValPairs = [
            (Vrt0, "vrt(v0,z,a)"),
            (Pe0,  "pe(v0,z,a)"),
            (Vrt1, "vrt(v1,a-z,a)"),
            (Pe1,  "pe(v1, a-z, a)")
        ]

        # find the model parameters using an arbitrary starting point:
        EZ2({"v0": .156, "v1" :.1144, "z":.0631, "a": .1263}, ObsValPairs)

        # the result should be: "v0" ≈ 0.1, "v1" ≈ 0.15, "z" ≈ 0.08, and "a" ≈ 0.12
    """
    import numpy as np
    from scipy import optimize

    def func(par, par_names, ObsValPairs):
        """
        Evaluates the model expression in ObsValPairs in the parameters par,
        and returns the differences with corresponding observations as a list. 
        """
        import numpy as np
        p = dict(zip(par_names, par))

        objfval = []
        for observed, model_expr in ObsValPairs:
            predicted = eval(model_expr, globals(), p)
            error = predicted - observed
            objfval.append(error)

        return objfval

    par = list(p_start.values())
    par_names = p_start.keys()
    fit = optimize.root(func, x0=par, args=(par_names, ObsValPairs), **kwargs)
    
    p_end = dict(zip(list(p_start.keys()), list(fit.x)))
    
    return p_end, fit



def batch(pstart, column_models, data, nrestart = 1, diagn=True, **kwargs):
    """
    Batch EZ2 model fitting

    Description:

         Fit the a simplified diffusion model for response time and
         accuracy in batch to each row of a data frame.

    Usage:

         batch(pstart, column_models, data, nrestart = 4, **kwargs)

    Arguments:

      pstart: Dictionary with model parameter starting values (floats).
              (Starting values are good guesses of the true value of the 
              parameters.)

    column_models: List of len(column_models) = len(data.columns) expressions 
              as strings that express the observed values of the means, variances, 
              proportions in the columns of data in terms of model parameters. 
              See examples below.


        data: Data frame with observed values of moments (RT means, 
              variances, proportions response errors) for different
              experimental conditions.

    nrestart: Number of times the fitting should restarted. Defaults to 4.

         ...: Named arguments passed onto scipy.optimize.minimize.

    Details:

         For each row in `data` the system of Method of Moments equations 

             value in `data` column j = value of expression in `column_models[j]`

         are solved for the model parameters used in the `column_models` 
         expressions and defined in `pstart`. 

         Solving is achieved by minimizing the _total discrepancy_ between the 
         values in the columns of `data` and the values predicted by the 
         expressions in `column_models`.

         The number of restarts needs to be increased if the model entails
         the same number of estimated parameters as the number of modeled
         moments and any of the values in the ‘fval’ column in the
         returned data frame is not close to ‘1.0e-08’.

    Value:

         DataFrame with parameter estimates for the column models fitted
         to each row in `data`. In addition, convergence information 
         about the solving the equations is given for each row:

           fval: The value of the total discrepancy between values in 
                `data` and the value predicted from the model expression.

          niter: The number of iterations taken by the solving algorithm.

         status: Integer that reports the convergence state of the 
                 algorithm. See help(scipy.optimize.OptimizeResult) 
                 for details. 

         succes: Bool. True implies succesful convergence. A very small
                 `fval` is much more important than `status` = True.

         norm_jac: Euclidean norm of the gradient. Should be 0 in theory.
                 Values larger than 1.0 may indicate improper solution
                 for that row, especially with non-small values of `fval`.
                 Try to recompute such cases for better solutions.


    Author(s):

         Raoul P. P. P. Grasman

    References:

         Ratcliff, R. (1978). Theory of Memory Retrieval. _Psychological
         review_ vol. 85 (2) pp. 59-108

         Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J.
         (2007). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation, _J.
         Math. Psych._ 53: 55-68.

    See Also:

         ‘v1’

    Examples:

         import pandas as pd

         ## create some data (theoretical values, not simulated) for a typic
         ## 2AFC experiment — in this example a lexical decision task
         ## (Needless to say, in reality you would moments computed from real data!)

         # true parameter values (10 different cases)
         par_df = pd.DataFrame({
             "v0": [0.1 + (0.3-0.1)*i/10 for i in range(11)],
             "v1": [0.15 + (0.4-0.15)*i/10 for i in range(11)],
             "z":  [0.15 + 0.03*(i-5)/5 for i in range(11)],
             "a":  [0.25]*11
         })

         # compute the theoretical variance (vrt0) and proportion error (pe0) for 
         # the 'word' response times, and the theoretical variance (vrt1) and error
         # (pe1) for the 'non-word' response times.
         dat_df = pd.DataFrame({
            'vrt0': eval('vrt(v0,z,a)', globals(), par_df),
            'pe0' : eval('pe(v0,z,a)', globals(), par_df),
            'vrt1': eval('vrt(v1,a-z,a)', globals(), par_df),
            'pe1' : eval('pe(v1,a-z,a)', globals(), par_df)
         })
         dat_df          # now pretend that `dat_df` is the data frame that 
                         # you have computed from real data; each row containing
                         # vrt0, pe0, vrt1, and pe1 from a single participant

         ## recover the parameters from the pretend data `dat_df`

         # specify the model expressions for each column
         column_models = [
             'vrt(v0,z,a)',        # first column: vrt0
             'pe(v0,z,a)',         # second column: pe0
             'vrt(v1,a-z,a)',      # third column: vrt1, starting point = a-z
             'pe(v1, a-z, a)']     # fourth column: pe1

         # solve for parameters: try 16 random starting values for each row
         pstart = {'v0': 0.17, 'v1': 0.15, 'z': 0.12, 'a': 0.25}

         random.seed(11)
         ez2fit = batch(pstart, column_model, dat_df, nrestart=16, tol=1e-15)
         ez2fit

    """
    from random import uniform
    import warnings
    import pandas as pd

    dat_df = data
    res = []
    resfit = []
    for i in range(len(dat_df)):
        
        ObsValPair = list(zip(dat_df.loc[i], column_models))
        p_end = pstart
        fit_end = None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for _ in range(nrestart):
                
                p_start = {parname: value * (1+uniform(-0.1,0.1)) for parname, value in pstart.items()}
                p_tmp, fit = EZ2(p_start, ObsValPair, **kwargs)
                
                if fit.success:
                    p_end = p_tmp
                    fit_end = fit
            
            fit_diagn = {"fval":fit.fun, "niter":fit.nfev, "status": fit.status, 
                         "success": fit.success, "message": fit.message,
                         "norm_error": sum(fit.fun**2,0)**0.5}
            
            if diagn:
                p_end.update(fit_diagn)

            res.append(p_end)
        
    return pd.DataFrame(res) 
