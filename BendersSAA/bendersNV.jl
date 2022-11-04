using JuMP, Plots, HiGHS, PlotlyJS
plotlyjs()

d = collect(60:1:150)
N = length(d)
p = (1/N)*ones(N)

# Original recourse function for purpouses of evaluation and plotting
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

g(x) = sum(p[s]*Q(x,d[s]) for s = 1:N)

f(x) = 10*x + g(x)

Plots.plot(g, collect(50:10:200))


# Auxillary recourse function to get the dual of X
function Q_aux(x, d)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variables(m, 
    begin
        y ≥ 0
        w ≥ 0
        x̃
    end)

    @constraints(m,
    begin
        con, x̃ == x
        y ≤ d
        y + w ≤ x̃
    end)

    @objective(m, Min, -25 * y - 5 * w)
    optimize!(m)
    return objective_value(m), shadow_price(con)
end

# Function to compute the cut of the iteration of benders decomposition
function compute_cut(x̂)
    x̂ = value(x)
    Q̄ = 0
    π̄ = 0
    for s = 1:N
        Q̄ += Q_aux(x̂, d[s])[1]
        π̄ += Q_aux(x̂, d[s])[2]
    end
    Q̄ = Q̄/N
    π̄ = π̄/N
    return (x -> Q̄ + π̄ * (x - x̂))
end


Q̲ = -5000
# Define master problem
master = Model(HiGHS.Optimizer)
set_silent(master)
@variables(master, begin
    0 <= x <= 200
    θ ≥ Q̲
end)
@objective(master, Min, 10*x + θ)
optimize!(master)

LB = objective_value(master)
x̂ = value(x)
vline!([x̂])
UB = f(x̂)
bestUB = UB

while (bestUB-LB) > 0.1
    l = compute_cut(x̂)
    @constraint(master, θ ≥ l(x))
    optimize!(master)
    LB = objective_value(master)
    x̂ = value(x)
    UB = f(x̂)
    bestUB = min(bestUB, UB)
    GAP = (bestUB-LB)
    @show bestUB, LB, GAP
    Plots.plot!(x -> 10*x + l(x), collect(50:10:200))
    vline!([x̂])
end

#Run current() to obtain the lines plotted during the execution of the loops
current()

# Final result of the amount of newspapers to buy came close to the optimal value obtained in the determinsitic equivalent, which is 128. (Benders achieved 128.5)
value(x)