The codes presented here are additional analysis of Figure 1B and 1C in our recent [Current Biology paper](http://www.cell.com/current-biology/fulltext/S0960-9822(16)30605-4)  

<div>
    <a href="https://plot.ly/~laoj/2/" target="_blank" title="Happy vs Fear" style="display: block; text-align: center;"><img src="https://plot.ly/~laoj/2.png" alt="Happy vs Fear" style="max-width: 100%;width: 600px;"  width="300" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="laoj:2"  src="https://plot.ly/embed.js" async></script>
    <a href="https://plot.ly/~laoj/10/" target="_blank" title="Novel Face vs Familiarized Face" style="display: block; text-align: center;"><img src="https://plot.ly/~laoj/10.png" alt="Novel Face vs Familiarized Face" style="max-width: 100%;width: 600px;"  width="300" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="laoj:10"  src="https://plot.ly/embed.js" async></script>
</div>

See also the Supplemental Information of the paper (Visual preference analysis of the *test* phase) for more details.  

Different analysis approaches are applied:  
0, The actual analysis code as appear in the paper: `Culture_Bady_testphase.m`. It applies Frequentist approach mainly using Bootstrap (for the figure) and Generalized linear mixed model (for the NHST)  
1, Surface mapping using Gaussian Process (`GPML_fit.m` and `GPML_ipython.ipynb`) or Generalized Additive Model (`GAM_fit.R`)  
2, Bayesian approach in PyMC (version 2 and 3, `dirichlet_fit_pymc.ipynb`) or JAGS (`rJAGS_ipython.ipynb`)  
