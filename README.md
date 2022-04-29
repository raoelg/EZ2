# EZ2

Updated Python module implementation of the R package "EZ2" that accompanies the paper

> Grasman, R. P. P., Wagenmakers, E.-J., & van der Maas, H. L. J. (2007). On the mean and variance of response times under the diffusion model with an application to parameter estimation, _J. Math. Psych._ 53: 55-68.

![EZ diffusion model for response times](https://media.springernature.com/lw685/springer-static/image/art%3A10.3758%2Fs13423-016-1081-y/MediaObjects/13423_2016_1081_Fig1_HTML.gif?as=webp)

### Example


Consider the example of a lexical decision 2AFC response time task from the paper: There are 'word' and 'non-word' stimuli presented to subjects who have to indicate as quickly as possible whether the stimulus is a word or a non-word for many trials. As a result of such an experiment, for each subject we have

- a mean response time for 'words' (`mrt0`) and a mean response time for 'non-words' (`mrt1`)
- a variance of the response times for 'words' (`vrt0`) and a variance of the response times for 'non-words' (`vrt1`)
- a percentage of trials in which the subject made errors for 'words' (`pe0`), and such a percentage for 'non-words' (`pe1`)

To estimate the parameters the _EZ2 way_ (statistically known as _Methods of Moments_ estimation), it turns out only the variances and error percentages are needed. 

The `EZ2` package is a lot more flexible than being applicable to just the above 2AFC type tasks. Effects of different types of stimuli may be hypothesized to lie in the drift rate, starting point, or boundary separation, or even the non-decision time component. That's why the model needs to be specified in terms of model equations in terms of relevant parameters. 

For the lexical decision task example it is often hypothesized that the drift rate for 'words', denoted _v<sub>0</sub>_, is higher than the drift rate for non-words, _v<sub>1</sub>_. The carefulness of the subject is reflected in the boundary seperation parameter _a_, which is usually assumed to be constant across randomized trials (at least in the short run of the experiment). If 'words' stimuli occur more frequently than 'non-words' stimuli, subjects tend to be biased to anticipate a 'words' stimulus more than a 'non-words' stimulus, resulting in more error responses on 'non-words'. This is modeled as a shift of the starting point, often denoted _z_, to the boundary that corresponds to the 'words' response, and the same amount away from the 'non-words' boundary. If the starting point for the 'words' stimuli is given by _z_, the starting 'non-words' is therefore then given by _a-z_. The equations needed for this lexical decision task are therefore given by

- `vrt0` is modeled as `EZ2.vrt(v0, z, a)`
- `vrt1` is modeled as `EZ2.vrt(v1, a-z, a)`
- `pe0` is modeled as `EZ2.pe(v0, z, a)`
- `pe1` is modeled as `EZ2.pe(v1, a-z, a)`

#### Generate data

First create some fake data. Data generated here are *population* variances and percentages of error responses for a number of sets of diffusion model parameter values. (This could correspond to different subjects having different values of _v<sub>0</sub>_, _v<sub>1</sub>_, _z_, and _a_.) In practice you would use of course _estimated_ variances and error percentages for each subject from the recorded response times for each subject.


```python
import pandas as pd
import EZ2

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
dat_df        # now pretend that `dat_df` is the data frame that 
              # you have computed from real data; each row containing
              # vrt0, pe0, vrt1, and pe1 from a single participant
```

The parameters and data look like this:

```python
print(pd.concat([par_df,dat_df],axis=1))
```

|    |   v0 |    v1 |     z |    a |      vrt0 |         pe0 |      vrt1 |        pe1 |
|---:|-----:|------:|------:|-----:|----------:|------------:|----------:|-----------:|
|  0 | 0.1  | 0.15  | 0.12  | 0.25 | 0.631635  | 0.0845497   | 0.283616  | 0.0196997  |
|  1 | 0.12 | 0.175 | 0.126 | 0.25 | 0.456897  | 0.0462423   | 0.203525  | 0.0128801  |
|  2 | 0.14 | 0.2   | 0.132 | 0.25 | 0.326752  | 0.0239326   | 0.149945  | 0.00887018 |
|  3 | 0.16 | 0.225 | 0.138 | 0.25 | 0.232702  | 0.0117509   | 0.113401  | 0.00646083 |
|  4 | 0.18 | 0.25  | 0.144 | 0.25 | 0.165954  | 0.00548281  | 0.087874  | 0.00498789 |
|  5 | 0.2  | 0.275 | 0.15  | 0.25 | 0.11902   | 0.00243346  | 0.0695974 | 0.00408571 |
|  6 | 0.22 | 0.3   | 0.156 | 0.25 | 0.0860831 | 0.00102804  | 0.0561972 | 0.00355256 |
|  7 | 0.24 | 0.325 | 0.162 | 0.25 | 0.0628774 | 0.000413546 | 0.0461548 | 0.00327962 |
|  8 | 0.26 | 0.35  | 0.168 | 0.25 | 0.0463874 | 0.000158435 | 0.0384793 | 0.00321474 |
|  9 | 0.28 | 0.375 | 0.174 | 0.25 | 0.034533  | 5.7814e-05  | 0.0325108 | 0.00334596 |
| 10 | 0.3  | 0.4   | 0.18  | 0.25 | 0.0258981 | 2.00936e-05 | 0.0278019 | 0.00369786 |


#### Run EZ2

```python
## recover the parameters from the pretend data `dat_df`

 # specify the model expressions for each column
column_models = [
     'EZ2.vrt(v0,z,a)',        # first column: vrt0
     'EZ2.pe(v0,z,a)',         # second column: pe0
     'EZ2.vrt(v1,a-z,a)',      # third column: vrt1, starting point = a-z
     'EZ2.pe(v1, a-z, a)']     # fourth column: pe1

# solve for parameters: try 16 random starting values for each row
pstart = {'v0': 0.17, 'v1': 0.15, 'z': 0.12, 'a': 0.25}

random.seed(11)
ez2fit = EZ2.batch(pstart, column_model, dat_df, nrestart=16, tol=1e-15)
ez2fit
```

The result looks like this:

|    |        v0 |       v1 |        z |        a |        fval |   niter |   status | success   |    norm_jac |
|---:|----------:|---------:|---------:|---------:|------------:|--------:|---------:|:----------|------------:|
|  0 | 0.0999994 | 0.15     | 0.120001 | 0.249999 | 4.064e-07 |      17 |        2 | False     | 0.0763213   |
|  1 | 0.12      | 0.175    | 0.126    | 0.25     | 1.608e-07 |      13 |        2 | False     | 0.150949    |
|  2 | 0.14      | 0.2      | 0.132    | 0.249999 | 6.837e-08 |       8 |        2 | False     | 0.335807    |
|  3 | 0.16      | 0.225    | 0.138    | 0.25     | 1.161e-08  |      14 |        2 | False     | 0.163066    |
|  4 | 0.18      | 0.25     | 0.144    | 0.25     | 2.368e-08 |      12 |        2 | False     | 0.044371    |
|  5 | 0.2       | 0.274999 | 0.149998 | 0.249999 | 2.586e-08 |      16 |        2 | False     | 3.60527e-08 |
|  6 | 0.220001  | 0.299996 | 0.155994 | 0.249995 | 3.569e-08 |      19 |        2 | False     | 0.0854866   |
|  7 | 0.240002  | 0.324993 | 0.16199  | 0.249993 | 9.119e-08 |      23 |        2 | False     | 0.0177482   |
|  8 | 0.260011  | 0.349955 | 0.167936 | 0.249947 | 2.709e-07 |      12 |        2 | False     | 0.013834    |
|  9 | 0.28004   | 0.374835 | 0.173783 | 0.249818 | 9.650e-07  |      31 |        2 | False     | 0.00198474  |
| 10 | 0.300071  | 0.399707 | 0.179626 | 0.249677 | 3.900e-06 |      30 |        2 | False     | 0.00143761  |

Despite the `success` column indicating `False`, the small `fval` are near zero suggesting a proper solution was found. Comparison of columns `v0`, `v1`, `z` and `a` with the same colums from the true parameter values in the previous table, shows that the true parameter values are retrieved well.

See `help(batch)` or `help(EZ2)` for documentation of the available function.
