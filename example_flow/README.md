# Data overview

We generated 1000 random points as places within the rectangular region defined by $x\in\left[500,1500\right)$ and $y\in\left[1000,2000\right)$. The synthetic flow data is generated according to gravity model
    $$G_{ij}= k\frac{{P_i}^{\alpha}{{A_j}^{\gamma}}}{{d_{ij}^{\beta}}},$$
where $P\sim\rm{Lognormal}\left(3,1\right)$, $A\sim\rm{Lognormal}\left(3,1\right)$, $d_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}$.

![image](https://github.com/Xuechen0123/FlowAlloc_SI_placeemb/blob/main/img/Fig2_Synthetic_data.png)

# Data format

## Place data

`place_id, place_x, place_y, propulsiveness, attractiveness`

## Flow data

`origin_place_id, destination_place_id, flow volume`
