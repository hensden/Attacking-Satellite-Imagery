## Evasion Attacks on Resisc45
### Source: [ART by IBM](https://github.com/IBM/adversarial-robustness-toolbox) and [Armory by Twosixlabs](https://github.com/twosixlabs/armory/)


### Results and Visualizations: 
#### Adversarial Patch Attack: [Source](https://arxiv.org/abs/1712.09665)
- Patches with target Classes Beach and Overpass: <br />
   ![Beach](repo_samples/beach.png) ![Overpass](repo_samples/overpass.png) 

- Merged patches which is 16% area of the whole image. <br />
   ![](repo_samples/p3.png) ![](repo_samples/p2.png) ![](repo_samples/p1.png)


#### Universal Perturbation Attack: [Source](https://arxiv.org/abs/1610.08401) 
- A single perturbation filter / mask is applied to all images, target class is randomized. <br />
   ![](repo_samples/u3.png) ![](repo_samples/u2.png) ![](repo_samples/u1.png)
