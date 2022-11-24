# Calculation of Molecular Diffusivity by Maxwell Stefans Model
Molecular diffusivity is a critical paramter when it come to accurate modelling of mass transport processes. This repository hosts a data driven model, based on maxwell Stefans's approach.
The molecular diffusivity of _Thomson Seedless_ grapes are estimated by fitting a numerical model to experimental data of drying grapes at various air temperatures and air velocities.
In addition to the molecular diffusivity this model also estimates the external mass transfer coefficient for the process. The detailed decription of the descritization nad solution procedure can be found in [Kulkarni et al.](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cjce.24514)
The non-ideal diffusion is modelled by Non-Random Two Liquid (NRTL) Model as described by [Renon and Prausnitz](https://doi.org/10.1002/aic.690140124).

## Contents

### Schematics
<img src="https://user-images.githubusercontent.com/35555732/203823058-d3300789-6e86-4f39-acf1-074c3ab89366.png" width="500" height="500">

### Governing Equations
<img src="https://user-images.githubusercontent.com/35555732/203823233-083e461d-f0cc-48a7-97f4-0e725d3158b8.png" width="560" height="290">

### Decritization Procedure
<img src="https://user-images.githubusercontent.com/35555732/203823402-3374193f-ff90-4c0f-8d99-7764d7487959.png" width="900" height="500">


### Solution Algorithm
![Model7_SolutionProcedure](https://user-images.githubusercontent.com/35555732/203823556-15cb5cb8-1bf9-40cc-94ee-16f006c822b0.png)


### References
 1. [Effective Maxwell‐Stefan diffusion model of near ambient air drying validated with experiments on Thomson seedless grapes](https://doi.org/10.1002/cjce.24514) by Kulkarni et al.
 2. [Local compositions in thermodynamic excess functions for liquid mixtures]( https://doi.org/10.1002/aic.690140124) by Henri Renon & J. M. Prausnitz


