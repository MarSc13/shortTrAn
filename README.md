# shortTrAn
Is an approach to calculate the local diffusion coefficient and the local speed of the direct motion from a large number of short trajectories. The approach is based on the approach from Hoze et al (2012) and Hoze, Holcman (2014) and was adapted to our datasets. The results are depicted in diffusion and directed motion maps. Subsequently, the decision on the locally dominant motion type is made by our self-developed guideline and displayed in a supermap.

This analysis was used to study the VSG (variant surface gylcoprotein) diffusion in the relation to the flagellar pocket (FP) on the unicellular, eukaryotic parasite Trypanosoma brucei. The size of approx. 20 µm in length and 3 µm in width, as well as, the 3D shape of the trypanosomes limit the length of the gained trajectories. Therefore the most popular and commonly used method, to calculate the diffusion coefficient from the fit of the MSD, is not suitable for our datasets to gain robust statistics. 

This repository contains Jupyter notebooks, Python scripts and exemplary datasets.

Workflow:

1. <b>shortTrAn_loop.ipynb</b> (shortTrAn/Trajectory_analysis):  This notebook enables the iteration over several data sets. In brief, a unit transformation from pixel to nanometre is performed. Trajectories are split in one step events and for each the displacement is ascertained. The resulting displacements are interpreted both in the light of a diffusion process to calculate a local diffusion coefficient and in light of a directed motion model to calculate a local velocity. Localisations errors can be considered. Finally, the results of the subpixelation are binned to a superpixel scale and spatially filtered.
2. <b>outline_marker_signal.py & Highlighting_fp.py</b> (shortTrAn/Outline_markerROI) - OPTIONAL:  The first script extracts the outline from single-molecule localisations of a structural element. The second script enables the highlighting of an part of the outline and can be optionally performed.
3. <b>Create_Maps.ipynb</b> or <b>Create_Maps_Plus_Outline.ipynb</b> (shortTrAn/Creation_maps/without_outline or shortTrAn/Creation_maps/with_outline): Generation of the diffusion (ellipse plot) and directed motion (heat map, quiver plot, (relative standard error map)) maps. Localisations errors can be considered and the consideration has to be stated again to enbale the generation of the relative standard error map of a directed motion. In comparison to the first notebook, this second notebook allows the plotting of the outline of a structure element onto the generated maps.
4. <b>Generation_Supermap.ipynb</b> (shortTrAn/Supermaps): This jupyter notebook provides an guideline to decide whether diffusion or directed motion is locally predominant. The results are displayed in a so-called supermap.

Additional Jupyter notebooks are available in shortTrAn/Trajectory_analysis, which allow the determination of different input parameter (e.g. padding, binning, spatial filtering).
