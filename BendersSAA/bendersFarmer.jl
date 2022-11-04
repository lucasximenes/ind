using JuMP, Plots, HiGHS, PlotlyJS, LinearAlgebra

# Benders decomposition for the farmer problem

PoP = [238, 210, 0] # price of purchase
PsP = [170, 150, 36, 10] # price of sale
CoP = [150, 230, 260] # cost of planting
p2 = [1/3, 1/3, 1/3] # probability of each state
Ys = [2 2.4 16; 2.5 3 20; 3 3.6 24]


function Q(x, Y) # big Y stands for yield, and x is the land allocated to each crop
    m = Model(HiGHS.Optimizer)
    set_silent(m)
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

# Auxillary recourse function to get the dual of X
function Q_aux(x, Y)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variables(m, 
    begin
        y[1:3] ≥ 0
        w[1:4] ≥ 0
        x̃[1:3]
    end)

    @constraints(m,
    begin
        con, x̃[1:3] .== x[1:3]
        Y[1]*x̃[1] + y[1] - w[1] ≥ 200
        Y[2]*x̃[2] + y[2] - w[2] ≥ 240
        w[3] + w[4] ≤ Y[3]*x̃[3]
        w[3] ≤ 6000
    end)

    @objective(m, Min, PoP ⋅ y - PsP ⋅ w)
    optimize!(m)
    return objective_value(m), shadow_price.(con)
end

function f(x)
    return CoP ⋅ x + sum(p2[s]*Q(x, Ys[s, :]) for s = 1:3)
end

# Function to compute the cut of the iteration of benders decomposition
function compute_cut(x̂)
    x̂ = value.(x)
    Q̄ = 0
    π̄ = [0, 0, 0]
    for s = 1:3
        aux1, aux2 = Q_aux(x̂, Ys[s, :])
        @show aux2
        Q̄ += aux1
        π̄ += aux2 
    end
    Q̄ /= 3
    π̄ /= 3
    return (x -> Q̄ + π̄  ⋅ (x - x̂))
end

# Initial lower bound with no real intuition behind it, because, for some reason, when trying the initial lower bound as
# the objective value of the actual optimal solution, the algorithm would not converge to the solution.
Q̲ = -1e6

# Define master problem
master = Model(HiGHS.Optimizer)
set_silent(master)
@variables(master, begin
    x[1:3] ≥ 0
    θ ≥ Q̲
end)
@constraint(master, sum(x) ≤ 500)
@objective(master, Min, CoP ⋅ x + θ)
optimize!(master)


LB = objective_value(master)
x̂ = value.(x)

UB = f(x̂)
bestUB = UB

# while (bestUB-LB) > 0.1
    l = compute_cut(x̂)
    @constraint(master, θ ≥ l(x))
    optimize!(master)
    LB = objective_value(master)
    x̂ = value.(x)
    UB = f(x̂)
    bestUB = min(bestUB, UB)
    GAP = (bestUB-LB)
    @show bestUB, LB, GAP
# end


# The allocation of land to each crop, which converged to the solution obtained in the deterministic equivalent! (170, 8)
value.(x)