#### EZ2
#### author: Raoul P. P. P. Grasman
#### License: Apache License 2.0

from scipy import optimize
import numpy as np
import pandas as pd
import random


def cmrt(nu, z, a, Ter=0, s = 0.1, GRAD=False):
    """
    Compute exit (decision) time mean and variance _conditional_ on exit point (chosen
    alternative) of a diffusion process

    Description:

         Given a boundary separation, a starting point, and a drift rate,
         this function computes the mean exit time/exit time variance of a
         one dimensional diffusion process under constant drift on an
         interval with absorbing boundaries, conditioned on the point of
         exit. Used as a model of information accumulation, it is gives the
         mean decision time/decision time variance of responses in a
         speeded two-alternative forced choice (2AFC) response time task,
         conditional on what alternative was decided upon.

    Usage:

         EZ2.cmrt(nu, z, a, s = 0.1, Ter=0)
         EZ2.cvrt(nu, z, a, s = 0.1, Ter=0)

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

         ‘EZ2.cmrt’ returns the mean exit/decision time(s) ‘EZ2.cvrt’    
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

         ‘EZ2-package’, ‘EZ2.mrt’, ‘EZ2.vrt’, ‘EZ2.batch’

    Examples:

             EZ2.mrt( 0.1, 0.08, 0.12, Ter=0.25)
             EZ2.vrt( 0.1, 0.08, 0.12)
    
    
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

    see help(EZ2.cmrt) for details.
    
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

           EZ2.mrt(nu, z, a, s = 0.1, Ter)
           EZ2.vrt(nu, z, a, s = 0.1)

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

         ‘EZ2.mrt’ returns the mean exit/decision time(s) ‘EZ2.vrt’ returns
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

         ‘EZ2-package’, ‘EZ2.cmrt’, ‘EZ2.cvrt’, ‘EZ2.pe’, ‘EZ2.batch’

    Examples:

         EZ2.mrt( 0.1, 0.08, 0.12, Ter=0.25)
         EZ2.vrt( 0.1, 0.08, 0.12)


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

    See help(EZ2.mrt) for details.

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

         EZ2.pe(nu, z, a, s = 0.1)

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

         ‘EZ2.pe’ returns the exit probability through lower end of the
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

         ‘EZ2-package’, ‘EZ2.cmrt’, ‘EZ2.cvrt’, ‘EZ2.mrt’, ‘EZ2.vrt’

    Examples:

         EZ2.pe(.1, .08, .12)

     
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







def data2ez(Pc, VRT, MRT = None, s = 0.1):
    """Convert observed sample moments to diffusion parameter values

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

         Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. P. P. 
         (2007). An EZ-diffusion model for response time and accuracy. 
         Psychonomic Bulletin & Review, 14(1), 3-22.
         
         Wagenmakers, E. J., van der Maas, H. L., Dolan, C. V., & Grasman, 
         R. P. P. P. (2008). EZ does it! Extensions of the EZ-diffusion 
         model. Psychonomic Bulletin & Review, 15(6), 1229-1235.

         Grasman, R.P.P.P., Wagenmakers, E.-J.& van der Maas, H.L.J.
         (2009). On the mean and variance of response times under the
         diffusion model with an application to parameter estimation.
         Journal of Mathematical Psychology, 53(2), 55-68.

    See Also:

         ‘EZ2.batch’

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



def EZ2_objf_oud(p, ObsValPairs, grad=False):
    
    objfval = 0.0
    objgrad = 0.0
    for pair in ObsValPairs:
        observed, model_expr = pair.split("~")
        observed = eval(observed, p)
        predicted, pred_grad = eval(model_expr, p)
        error = predicted - observed
        objfval += 0.5 * error**2
        if grad:
            objgrad = objgrad + error * np.array(list(pred_grad[0]))
            
    return (objfval, objgrad) if grad else (objfval, None)

def EZ2_objf(par, par_names, ObsValPairs):
    """
    Implements an objective function for finding the parameter values for solving the
    Method of Moments equations. The objective function is simply the squared sum of the
    deviations. This makes it possible to fit an underparameterized model using least
    squares.
    
    This function is supposed to be private.
    
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


def EZ2_objgrad(par, par_names, ObsValPairs):
    """
    DO NOT USE.
    
    This function is supposed to compute the gradients of the objective function. However, due to the
    lack of symbolic derivatives computation in Python, this is only correct if the Value expressions 
    (in terms of EZ_<mrt,vrt,pe>(nu,z,a,[Ter]) do not contain function of the parameter defined by 
    `par_names`. That is `EZ_vrt(nu,a-z,a)` would make the gradient computed by this function invalid.
    Yet being able to use such expressions is crucial. Hence, for now, this function is not used for
    minimizing the objective function EZ2_objf.
    
    Not made to work; let alone tested.
    """
    import numpy as np
    p = dict(zip(par_names, par))
    
    objfval = 0.0
    objgrad = 0.0
    for observed, model_expr in ObsValPairs:
        predicted, grad = eval(model_expr, globals(), p)
        error = np.array(predicted - observed)
        objgrad += error.T @ np.array(list(pred_grad[0])) * 1e6
            
    return objgrad

    
    
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
              "EZ2_pe(v1, z, a)") or (vrt2, "EZ2_pe(v2, a-z, a)") if `vrt2`
              exists in the global environment and is numeric.

         ...: More arguments that are passed to `scipy.optimize.minimize`.

    Details:

         EZ2 fits a simplified version of the diffusion model for human and
         monkey response times and accuracy to the means and variances of
         the observables. This model of information accumulation and
         decision is a simplified version of Ratcliff's diffusion model
         (1978).

         Use ‘EZ2batch’ for more user friendly fitting automatically each
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

         ‘EZ2-package’, ‘EZ2batch’

    Examples:

        # An example with theoretical population values as 'observations'

        # generate observations:
        Vrt0 = EZ2_vrt(0.1, 0.08, 0.12)        # RT variance 'word' responses
        Pe0  = EZ2_pe(0.1, 0.08, 0.12)         # Proportion errors 'word' response 
        Vrt1 = EZ2_vrt(0.15, 0.12-0.08, 0.12)  # RT variance 'non-word' responses
        Pe1  = EZ2_pe(0.15, 0.12-0.08, 0.12)   # Proportion errors 'non-word' responses

        # specify the model for the 'observations': "v0", "v1", "z", and "a" are
        # the names of the model parameters to be estimated by EZ2
        ObsValPairs = [
            (Vrt0, "EZ2_vrt(v0,z,a)"),
            (Pe0,  "EZ2_pe(v0,z,a)"),
            (Vrt1, "EZ2_vrt(v1,a-z,a)"),
            (Pe1,  "EZ2_pe(v1, a-z, a)")
        ]

        # find the model parameters using an arbitrary starting point:
        EZ2({"v0": .156, "v1" :.1144, "z":.0631, "a": .1263}, ObsValPairs)

        # the result should be: "v0" ≈ 0.1, "v1" ≈ 0.15, "z" ≈ 0.08, and "a" ≈ 0.12
    """
    import numpy as np

    par = list(p_start.values())
    par_names = list(p_start.keys())
    fit = optimize.minimize(EZ2_objf, x0=par, args=(par_names, ObsValPairs), **kwargs)
    
    p_end = dict(zip(list(p_start.keys()), list(fit.x)))
    
    return p_end, fit





def batch(pstart, column_models, data, nrestart = 4, **kwargs):
    """
    Batch EZ2 model fitting

    Description:

         Fit the a simplified diffusion model for response time and
         accuracy in batch to each row of a data frame.

    Usage:

         EZ2.batch(pstart, column_models, data, nrestart = 4, **kwargs)

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

         ‘EZ2’

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
            'vrt0': eval('EZ2.vrt(v0,z,a)', globals(), par_df),
            'pe0' : eval('EZ2.pe(v0,z,a)', globals(), par_df),
            'vrt1': eval('EZ2.vrt(v1,a-z,a)', globals(), par_df),
            'pe1' : eval('EZ2.pe(v1,a-z,a)', globals(), par_df)
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
         ez2fit = EZ2.batch(pstart, column_models, dat_df, nrestart=16, tol=1e-15)
         ez2fit

    """

    from random import uniform
    import warnings
    import pandas

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
                p_tmp, fit = EZ2(p_start, ObsValPair, **kwargs)
                
                if fit.fun < fval:
                    p_end = p_tmp
                    fval = fit.fun
                    
            
            fit_diagn = {"fval":fit.fun, "niter":fit.nit, "status": fit.status, 
                         "success": fit.success, "norm_jac": sum(fit.jac**2,0)**0.5}
            p_end.update(fit_diagn)
        
            res.append(p_end)
        
    return pd.DataFrame(res) 
