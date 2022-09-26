using JuMP, Plots, HiGHS, LinearAlgebra

#=
Every morning, the newsboy goes to the editor of the newspaper and buys a quantity of x newspapers at a cost of c per unit. This x
amount is limited upwards by a u value, as the newsboy has finite purchasing power. The decision is made under uncertainty since the
demand D newspapers of the day is unknown. The number of newspapers y is sold at a price of q per unit. The newsboy also has an
agreement with the newspaper editor: the number w of unsold newspapers can be returned to the editor, who pays a r price for him.
=#

d = collect(60:1:150) # Demand vector [50,150]
N = length(d)
p = (1/N).*ones(N,1)

# Q would be the second stage problem, where you receive the amount of newspaper bought and calculate how much you make by
# selling and returning the newspapers, given a demand d.
function Q(x,d)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variables(m, 
    begin
        y ≥ 0
        w ≥ 0
    end)

    @constraints(m,
    begin
        y ≤ d
        y + w ≤ x
    end)

    @objective(m, Min, -25 * y - 5 * w)
    optimize!(m)
    return objective_value(m)
end

f(x) = 10*x + sum(p[s]*Q(x,d[s]) for s = 1:N) # This would be the first stage problem, where you decide how many newspapers to buy, abstracting the second stage 
#using Q(x, d).

Plots.plot(f, collect(50:10:200))
# Turning the problem into a deterministic one by enumerating all the cases of demand d, then solving it.

m = Model(HiGHS.Optimizer)
set_silent(m)
@variables(m,
begin
    0 ≤ x ≤ 150
    y[1:N] ≥ 0
    w[1:N] ≥ 0
end)

@constraints(m,
begin
    ct1[s=1:N], y[s] ≤ d[s]
    ct2[s=1:N], y[s] + w[s] ≤ x
end)

@objective(m, Min, 10 * x + sum(p[s]*(-25*y[s] -5*w[s]) for s = 1:N))

optimize!(m)

value(x) # Optimal amount of newspapers to buy, in this case, 130.
objective_value(m)
# Calculating value of stochastic solution

g(x) = 10*x + Q(x, 100) # Here we calculate the objective value of the deterministic version of the problem, where we use the the demand as the 
# average of the demand vector. There is no need to minimize the first stage problem, since we already know the optimal amount of newspapers to buy (the avg of demand).

g(100) # EEV, the cost that was expected when using the mean of the distribution as the "deterministic" demand
f(100) # The cost that was actually incurred when using the mean as demand

VSS = f(100) - f(130)# Value of the stochastic solution

# Calculating value of perfect information

# In the case of perfect information, we will always buy the amount of newspaper equal to the demand, so we know that for every
# newspaper bought, we will receive (q - c) revenue. The cost of buying the newspaper is c, so the profit is (q - c) * x.

PI = sum(p[s]*-(25 - 10) * d[s] for s = 1:N) # Perfect information
EVPI = f(130) - PI# Expected value of perfect information

#=
João is a farmer from a small town who specializes in growing wheat, corn, and sugar beet. He owns 500 km2 of land and must decide
the amount of land to be allocated to each of the crops. João faces several restrictions regarding his planting. First, he must have at least
200 Tons (T) of wheat and 240 T of corn to feed his cattle. Such quantities can be obtained through own plantation or by buying from
the city's cooperative. The purchase prices per ton of wheat are 238 R$/T and per ton of corn 210 R$/T. On the other hand, any excess
produced in relation to the minimum can be sold at the cooperative, however with a 40% discount on the purchase price (170 R$/T for
wheat and 150 R$/T for corn) per the cooperative's margin account. Another important restriction concerns the sale of sugar beet. By
law, the sale price of a ton of beet at the cooperative is fixed at 36 R$/T for the first 6000 T sold. After this amount, the sale price
becomes 10 R$/T. In addition, the cooperative where João trades his products does not have sugar beet for purchase.
Let's assume that the planting cost of each crop is: 150 R$/km2 for wheat, 230 R$/km2 for corn, and 260 R$/km2 for sugar beet. The
uncertainty of the problem lies in the productivity of the land. João does not know a priori how much each km2 of land will yield in tons
of culture. Now assume that three scenarios of equal probabilities of occurrence were sampled: "good", "average" and "poor". In each of
the states, the yield of each crop is given by:

1. "Bad" state: Wheat = 2 T/km2; Corn = 2.4 T/km2; Beetroot = 16 T/km2;
2. "Average" state: Wheat = 2.5 T/km2; Corn = 3.0 T/km2; Beetroot = 20 T/km2;
3. "Good" state: Wheat = 3 T/km2; Corn = 3.6 T/km2; Beetroot = 24 T/km2;

=#

PoP = [238, 210, 0] # price of purchase
PsP = [170, 150, 36, 10] # price of sale
CoP = [150, 230, 260] # cost of planting
p2 = [1/3, 1/3, 1/3] # probability of each state
Ys = [2 2.4 16; 2.5 3 20; 3 3.6 24] # yield of each crop in each state

# Second stage problem, where you receive the amount of land allocated to each crop and calculate the money spent and earned, given the yield of the land.
function Q2(x, Y) # big Y stands for yield, and x is the land allocated to each crop
    m = Model(HiGHS.Optimizer)
    @variables(m, 
    begin
        y[1:3] ≥ 0
        w[1:4] ≥ 0
    end)

    @constraints(m,
    begin
        Y[1]*x[1] + y[1] - w[1] ≥ 200
        Y[2]*x[2] + y[2] - w[2] ≥ 240
        w[3] + w[4] ≤ Y[3]*x[3]
        w[3] ≤ 6000
    end)

    @objective(m, Min, PoP ⋅ y - PsP ⋅ w)
    optimize!(m)
    return objective_value(m)
end

# Firt stage problem, where you decide how much land to allocate to each crop, abstracting the second stage using Q2(x, Y).
function f2(x)
    return CoP ⋅ x + sum(p2[s]*Q2(x, Ys[s, :]) for s = 1:3)
end

# Turning the problem into a deterministic one by enumerating all the cases of yield Y, then solving it.

m2 = Model(HiGHS.Optimizer)

@variables(m2, 
begin
    x[1:3] ≥ 0
    y[1:3, 1:3] ≥ 0
    w[1:3, 1:4] ≥ 0
end)

@constraints(m2,
begin
    ct1[s=1:3], Ys[s, 1]*x[1] + y[s, 1] - w[s, 1] ≥ 200
    ct2[s=1:3], Ys[s, 2]*x[2] + y[s, 2] - w[s, 2] ≥ 240
    ct3[s=1:3], w[s, 3] + w[s, 4] ≤ Ys[s, 3]*x[3]
    ct4[s=1:3], w[s, 3] ≤ 6000
    ct5, sum(x) ≤ 500
end)

@objective(m2, Min, CoP ⋅ x + sum(p2[s]*(PoP ⋅ y[s, :] - PsP ⋅ w[s, :]) for s = 1:3))

optimize!(m2)
Stochastic_Allocations = value.(x)

# Calculating value of stochastic solution

# Adaptating Q2 to transform it into the deterministic problem by including x (the crop allocations) as a decision variable,
# and including it in the objective function.

function Q2_D(Y) 
    m = Model(HiGHS.Optimizer)
    @variables(m, 
    begin
        x[1:3] ≥ 0
        y[1:3] ≥ 0
        w[1:4] ≥ 0
    end)

    @constraints(m,
    begin
        x[1] + x[2] + x[3] ≤ 500
        Y[1]*x[1] + y[1] - w[1] ≥ 200
        Y[2]*x[2] + y[2] - w[2] ≥ 240
        w[3] + w[4] ≤ Y[3]*x[3]
        w[3] ≤ 6000
    end)

    @objective(m, Min, CoP ⋅ x + PoP ⋅ y - PsP ⋅ w)
    optimize!(m)
    return value.(x), objective_value(m)
end

EEV_Allocation, EEV_expected_value = Q2_D(Ys[2, :]) #We use the mean yield (second row on the yield matrix) to calculate the EEV. 

# EEV_expected_value is the value that the mean solution expected to obtain. EEV_expected_value - f2(EEV_Allocation) shows the difference between them

EEV_expected_value - f2(EEV_Allocation)
VSS = f2(EEV_Allocation) - f2(Stochastic_Allocations)

# Calculating the Perfect Information values
PI = sum(p2[s]*Q2_D(Ys[s, :])[2] for s = 1:3)
# Calculating EVPI
EVPI =f2(Stochastic_Allocations) - PI