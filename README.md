# soccer-player-valuation

Companion code for [European Football Player Valuation: Integrating Financial Models and Network Theory](https://www.degruyter.com/document/doi/10.1515/jqas-2024-0006/html) ([preprint link](https://arxiv.org/pdf/2312.16179)) by Jimmy Risk and Albert Cohen

Contains three example notebooks:

* ``1_Futsal_Example.ipynb``: Goes through a minimalist futsal example to cover the metrics we developed in our paper in an illustrative way.  *(Matches Section 3.1 in the paper.)*
* ``2_EPL_Performance_Analysis.ipynb``: Goes into detail about the player performance process ($\pi$), including how it is obtained, aggregated, and has its parameters estimated.  *(Matches Section 5.1 in the paper.)*
* ``3_EPL_Financial_Analysis.ipynb``: Goes into detail about financial valuation of the players.  *(Matches Section 5.2 in the paper.)*

The examples utilize real world data for the English Premier League (EPL) spanning 2018--2023, specifically focusing on Liverpool *(Mohamed Salah, Trent Alexander-Arnold, Virgil van Dijk)*, Arsenal *(Eddie Nketiah, Granit Xhaka, Rob Holding)*, and Brighton *(Pascal Groß, Solly March, Lewis Dunk)*.  The code, however, can be used generally if one has the data.
